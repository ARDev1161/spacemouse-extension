# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import os
import json
from typing import Optional
from srl.spacemouse.spacemouse import SpaceMouse
from srl.spacemouse.spacemousefilter import SpaceMouseFilter
from srl.spacemouse.device import DEVICE_NAMES, DEVICE_SPECS
from omni.isaac.ui.ui_utils import setup_ui_headers, get_style
import numpy as np
import carb

import omni.ui as ui
from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription
from omni.kit.window.property.templates import LABEL_WIDTH
from omni.kit.window.extensions import SimpleCheckBox
import weakref

import omni.ext
import asyncio
from omni.isaac.core import World

from functools import partial

from srl.spacemouse.ui_utils import xyz_plot_builder, combo_floatfield_slider_builder,  multi_cb_builder, combo_cb_dropdown_builder
from srl.spacemouse.camera_settings import clear_camera_settings

instance = None
SPNAVCAM_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "spacemouse_viewport.py")
DEFAULT_SETTINGS = {
    "engage": False,
    "device_name": None,
    "auto_script": False,
    "mode_translation": True,
    "mode_rotation": True,
    "smoothing_factor": 0.5,
    "softmax_temp": 0.85,
    "translation_sensitivity": 1.0,
    "rotation_sensitivity": 1.0,
    "translation_deadband": 0.1,
    "rotation_deadband": 0.1,
}


def get_global_spacemouse() -> Optional[SpaceMouse]:
    return instance._device


def get_global_spacemouse_extension() -> Optional[object]:
    return instance


class SpaceMouseExtension(omni.ext.IExt):

    def on_startup(self, ext_id: str):
        self._ext_id = ext_id
        menu_items = [MenuItemDescription(name="SpaceMouse", onclick_fn=lambda a=weakref.proxy(self): a._menu_callback())]
        self._menu_items = menu_items
        add_menu_items(self._menu_items, "SRL")
        self._device = None
        self._models = dict()
        self._build_ui(
            name="SpaceMouse",
            title="SpaceMouse",
            doc_link="",
            overview="Provides a SpaceMouse class and manages configuration UI for a global singleton",
            file_path=os.path.abspath(__file__),
            number_of_extra_frames=2,
            window_width=350,
        )

        self._settings_path = self._get_user_settings_path()
        self._settings_data = self._load_user_settings(self._settings_path)
        self._suppress_setting_write = False
        self._reset_confirm_window = None
        self._script_globals = None
        self._auto_script_subscription = None
        self._auto_script_tries = 0
        frame = self.get_frame(index=0)
        self._models = {}
        self.build_control_ui(frame)
        self.build_data_ui(self.get_frame(index=1))
        # Read defaults straight from the models so that we start the state in sync with the UI
        self.filter = SpaceMouseFilter(
            self._models["Smoothing Factor"][0].get_value_as_float(),
            self._models["Softmax Temperature"][0].get_value_as_float(),
            self._models["Translation Sensitivity"][0].get_value_as_float(),
            self._models["Rotation Sensitivity"][0].get_value_as_float(),
            self._models["Translation Deadband"][0].get_value_as_float(),
            self._models["Rotation Deadband"][0].get_value_as_float(),
            self._models["Modes"][0].get_value_as_bool(),
            self._models["Modes"][1].get_value_as_bool()
        )
        self._plotting_event_subscription = None
        self._plotting_buffer = np.zeros((360, 6))
        self.engage_sub_handle = self._models["Engage"][0].subscribe_value_changed_fn(self._engage_value_changed)
        global instance
        instance = self
        self._auto_engage_if_requested()

    def _build_ui(self, name, title, doc_link, overview, file_path, number_of_extra_frames, window_width):
        self._window = omni.ui.Window(
            name, width=window_width, height=0, visible=True, dockPreference=ui.DockPreference.RIGHT_TOP
        )
        self._window.deferred_dock_in("Stage", ui.DockPolicy.TARGET_WINDOW_IS_ACTIVE)
        self._extra_frames = []
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                setup_ui_headers(self._ext_id, file_path, title, doc_link, overview)
                with ui.VStack(style=get_style(), spacing=5, height=0):
                    for i in range(number_of_extra_frames):
                        self._extra_frames.append(
                            ui.CollapsableFrame(
                                title="",
                                width=ui.Fraction(0.33),
                                height=0,
                                visible=False,
                                collapsed=False,
                                style=get_style(),
                                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                            )
                        )

    def build_control_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                frame.title = "Settings"
                frame.visible = True

                engage_default = bool(self._get_setting("engage", DEFAULT_SETTINGS["engage"]))
                device_default_name = self._get_setting("device_name", DEVICE_NAMES[0])
                device_default_index = 0
                if device_default_name in DEVICE_NAMES:
                    device_default_index = DEVICE_NAMES.index(device_default_name)

                dict = {
                    "label": "Engage",
                    "tooltip": "Connect and enable the choosen device",
                    "on_clicked_fn": [self._on_engage_event, self._on_device_selected],
                    "items": DEVICE_NAMES,
                    "default_val": [engage_default, device_default_index],
                }
                self._models["Engage"] = combo_cb_dropdown_builder(**dict)

                with ui.HStack():
                    ui.Label("Script", width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER)
                    ui.Button("SpacemouseCam", clicked_fn=self._run_spnavcam_script)
                    ui.Spacer(width=8)
                    auto_script_default = bool(self._get_setting("auto_script", DEFAULT_SETTINGS["auto_script"]))
                    auto_model = ui.SimpleBoolModel(default_value=auto_script_default)
                    SimpleCheckBox(auto_script_default, self._on_auto_script_changed, model=auto_model)
                    ui.Label("Auto Run", width=80, alignment=ui.Alignment.LEFT_CENTER)
                    self._models["Auto Script"] = auto_model


                trans_mode_default = bool(self._get_setting("mode_translation", DEFAULT_SETTINGS["mode_translation"]))
                rot_mode_default = bool(self._get_setting("mode_rotation", DEFAULT_SETTINGS["mode_rotation"]))
                dict = {
                    "label": "Modes",
                    "text": ["Translation", "Rotation"],
                    "count": 2,
                    "default_val": [trans_mode_default, rot_mode_default],
                    "on_clicked_fn": [partial(self._on_modes_event, "trans"), partial(self._on_modes_event, "rot")],
                }
                self._models["Modes"] = multi_cb_builder(**dict)

                dict = {
                    "label": "Smoothing Factor",
                    "tooltip": ["How much to weight historical signal against current signal. Higher values will consider the current signal less and less. `alpha` in an exponential weighted average of the control signal.", ""],
                    "default_val": float(self._get_setting("smoothing_factor", DEFAULT_SETTINGS["smoothing_factor"])),
                    "min": 0.0,
                    "max": 0.99
                }
                self._models["Smoothing Factor"] = combo_floatfield_slider_builder(**dict)
                self._models["Smoothing Factor"][0].add_value_changed_fn(self._on_smoothing_event)

                dict = {
                    "label": "Softmax Temperature",
                    "tooltip": ["How much to exagerate differences in the components of motion. Smaller values make the strongest component dominate, while larger values will be less and less different from the original input.", ""],
                    "default_val": float(self._get_setting("softmax_temp", DEFAULT_SETTINGS["softmax_temp"])),
                    "min": 0.01,
                    "max": 2.
                }
                self._models["Softmax Temperature"] = combo_floatfield_slider_builder(**dict)
                self._models["Softmax Temperature"][0].add_value_changed_fn(partial(self._on_softmax_event, "trans"))

                dict = {
                    "label": "Translation Sensitivity",
                    "tooltip": ["Multiplier applied to translation inputs", ""],
                    "default_val": float(self._get_setting("translation_sensitivity", DEFAULT_SETTINGS["translation_sensitivity"])),
                    "min": 0,
                    "max": 2.
                }
                self._models["Translation Sensitivity"] = combo_floatfield_slider_builder(**dict)
                self._models["Translation Sensitivity"][0].add_value_changed_fn(partial(self._on_sensitivity_event, "trans"))
                dict = {
                    "label": "Rotation Sensitivity",
                    "tooltip": ["Multiplier applied to rotation inputs", ""],
                    "default_val": float(self._get_setting("rotation_sensitivity", DEFAULT_SETTINGS["rotation_sensitivity"])),
                    "min": 0,
                    "max": 2.
                }
                self._models["Rotation Sensitivity"] = combo_floatfield_slider_builder(**dict)
                self._models["Rotation Sensitivity"][0].add_value_changed_fn(partial(self._on_sensitivity_event, "rot"))

                dict = {
                    "label": "Translation Deadband",
                    "tooltip": ["Threshold below which to zero translation inputs component wise", ""],
                    "default_val": float(self._get_setting("translation_deadband", DEFAULT_SETTINGS["translation_deadband"])),
                    "min": 0,
                    "max": .8
                }

                self._models["Translation Deadband"] = combo_floatfield_slider_builder(**dict)
                self._models["Translation Deadband"][0].add_value_changed_fn(partial(self._on_deadband_event, "trans"))
                dict = {
                    "label": "Rotation Deadband",
                    "tooltip": ["Threshold below which to zero rotation inputs component wise", ""],
                    "default_val": float(self._get_setting("rotation_deadband", DEFAULT_SETTINGS["rotation_deadband"])),
                    "min": 0,
                    "max": .8
                }
                self._models["Rotation Deadband"] = combo_floatfield_slider_builder(**dict)
                self._models["Rotation Deadband"][0].add_value_changed_fn(partial(self._on_deadband_event, "rot"))

                with ui.HStack():
                    ui.Label("Reset", width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER)
                    ui.Button("Reset Settings to Default", clicked_fn=self._open_reset_confirm)

        return

    def build_data_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                frame.title = "Data"
                frame.visible = True

                kwargs = {
                    "label": "XYZ",
                    "data": [[],[],[]],
                    "include_norm": False
                }
                self._models["xyz_plot"], self._models[
                    "xyz_vals"
                ] = xyz_plot_builder(**kwargs)

                kwargs = {
                    "label": "RPY",
                    "data": [[],[],[]],
                    "value_names": ("R", "P", "Y"),
                    "include_norm": False
                }
                self._models["rpy_plot"], self._models[
                    "rpy_vals"
                ] = xyz_plot_builder(**kwargs)

        return

    def toggle_plotting_event_subscription(self, val=None):
        if val:
            if not self._plotting_event_subscription:
                self._plotting_event_subscription = (
                    omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_plotting_step)
                )
            else:
                self._plotting_event_subscription = None
        else:
            self._plotting_event_subscription = None

    def _on_plotting_step(self, e: carb.events.IEvent):
        if self._device is None:
            return
        control = self._device.get_controller_state()
        if control is None:
            return
        self._plotting_buffer = np.roll(self._plotting_buffer, shift=1, axis=0)
        self._plotting_buffer[0, :3] = control.xyz
        self._plotting_buffer[0, 3:] = control.rpy
        for i in range(3):
            self._models["xyz_plot"][i].set_data(*self._plotting_buffer[:, i])
            self._models["xyz_vals"][i].set_value(self._plotting_buffer[0, i])

            self._models["rpy_plot"][i].set_data(*self._plotting_buffer[:, 3 + i])
            self._models["rpy_vals"][i].set_value(self._plotting_buffer[0, 3 + i])

        # Plotting norms can be helpful for checking whether the filtering is causing
        # changing the energy of the input too drastically
        if len(self._models["xyz_plot"]) == 4:
            self._models["xyz_plot"][3].set_data(*np.linalg.norm(self._plotting_buffer[:, :3], axis=1))
            self._models["xyz_vals"][3].set_value(np.linalg.norm(self._plotting_buffer[0,:3]))

        if len(self._models["rpy_plot"]) == 4:
            self._models["rpy_plot"][3].set_data(*np.linalg.norm(self._plotting_buffer[:, 3:], axis=1))
            self._models["rpy_vals"][3].set_value(np.linalg.norm(self._plotting_buffer[0,3:]))


    def get_frame(self, index):
        if index >= len(self._extra_frames):
            raise Exception("there were {} extra frames created only".format(len(self._extra_frames)))
        return self._extra_frames[index]

    def _menu_callback(self):
        self._window.visible = not self._window.visible
        return

    def _on_modes_event(self, kind, model):
        if kind == "trans":
            self.filter.translation_enabled = model
            self._set_setting("mode_translation", bool(model))
        elif kind == "rot":
            self.filter.rotation_enabled = model
            self._set_setting("mode_rotation", bool(model))

    def _on_sensitivity_event(self, kind, model):
        if kind == "trans":
            self.filter.translation_modifier = model.get_value_as_float()
            self._set_setting("translation_sensitivity", self.filter.translation_modifier)
        elif kind == "rot":
            self.filter.rotation_modifier = model.get_value_as_float()
            self._set_setting("rotation_sensitivity", self.filter.rotation_modifier)

    def _on_smoothing_event(self, model):
        self.filter.smoothing_factor = model.get_value_as_float()
        self._set_setting("smoothing_factor", self.filter.smoothing_factor)

    def _on_deadband_event(self, kind, model):
        if kind == "trans":
            self.filter.translation_deadband = model.get_value_as_float()
            self._set_setting("translation_deadband", self.filter.translation_deadband)
        elif kind == "rot":
            self.filter.rotation_deadband = model.get_value_as_float()
            self._set_setting("rotation_deadband", self.filter.rotation_deadband)

    def _on_softmax_event(self, kind, model):
        self.filter.softmax_temp = model.get_value_as_float()
        self._set_setting("softmax_temp", self.filter.softmax_temp)

    def _on_engage_event(self, model):
        cb_model, dropdown_model = self._models["Engage"]
        device_selection = dropdown_model.model.get_item_value_model().as_int
        device_selection = DEVICE_NAMES[device_selection]
        if self._device and self._device.is_running and cb_model.as_bool == False:
            asyncio.ensure_future(
                self._on_disengage_event_async()
            )

        elif cb_model.as_bool:
            # User (or code!) tried to make it true
            if self._device and self._device.is_running:
                return
            asyncio.ensure_future(
                self._on_engage_event_async(device_selection, cb_model)
            )

    async def discover_mouse(self):
        cb_model, dropdown_model = self._models["Engage"]
        device_selection = dropdown_model.model.get_item_value_model().as_int
        device_selection = DEVICE_NAMES[device_selection]
        if self._device and self._device.is_running:
            return True

        for i, device_selection in enumerate(DEVICE_NAMES):
            dropdown_model.model.get_item_value_model().set_value(i)
            engagement_result = await self._on_engage_event_async(device_selection, cb_model)
            if engagement_result:
                cb_model.set_value(True)
                return True
        dropdown_model.model.get_item_value_model().set_value(0)
        return False

    async def _on_engage_event_async(self, device_name, model):
        try:
            spec = DEVICE_SPECS[device_name]
            self._device = SpaceMouse(spec)
            self._device.set_position_callback(self.filter._translation_modifier)
            self._device.set_rotation_callback(self.filter._rotation_modifier)
            self._device.set_unexpected_close_callback(self._on_unexpected_close)
            self._device.run()
            if self._get_setting("auto_script", DEFAULT_SETTINGS["auto_script"]):
                self._request_auto_script_start()
            return True
        except RuntimeError:
            carb.log_error(f"Unable to open device { spec.name }. Did you plug in the device, set up spacenavd and udev rules correctly?")
            self._device = None
            model.set_value(False)
            return False

    def _on_unexpected_close(self):
        cb_model, dropdown_model = self._models["Engage"]
        self._device = None
        cb_model.set_value(False)

    async def _on_disengage_event_async(self):
        if self._device:
            self._device.close()
            self._device = None

    def _engage_value_changed(self, model):
        self.toggle_plotting_event_subscription(model.as_bool)
        self._set_setting("engage", bool(model.as_bool))

    def _on_device_selected(self, device_name):
        self._set_setting("device_name", device_name)

    def _on_auto_script_changed(self, model):
        enabled = bool(model)
        self._set_setting("auto_script", enabled)
        if not enabled:
            self._cancel_auto_script_start()

    def _request_auto_script_start(self):
        if self._auto_script_subscription is not None:
            return
        self._auto_script_tries = -10
        self._auto_script_subscription = (
            omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._auto_script_tick)
        )

    def _cancel_auto_script_start(self):
        if self._auto_script_subscription is not None:
            self._auto_script_subscription = None

    def _auto_script_tick(self, e: carb.events.IEvent):
        self._auto_script_tries += 1
        if self._auto_script_tries < 0:
            return
        if self._run_spnavcam_script(quiet=True):
            self._auto_script_subscription = None
            return
        if self._auto_script_tries > 600:
            carb.log_warn("Auto Run: failed to start script after multiple attempts.")
            self._auto_script_subscription = None

    def _get_setting(self, name: str, default):
        return self._settings_data.get(name, default)

    def _set_setting(self, name: str, value):
        self._settings_data[name] = value
        if not self._suppress_setting_write:
            self._save_user_settings(self._settings_path, self._settings_data)

    def _auto_engage_if_requested(self):
        try:
            if not self._get_setting("engage", DEFAULT_SETTINGS["engage"]):
                return
            cb_model, _ = self._models["Engage"]
            if not cb_model.as_bool:
                cb_model.set_value(True)
            self._on_engage_event(cb_model)
        except Exception as exc:
            carb.log_warn(f"Auto engage failed: {exc}")

    def _get_user_settings_path(self) -> str:
        tokens = carb.tokens.get_tokens_interface()
        user_config_dir = tokens.resolve("${user_config_dir}")
        return os.path.join(user_config_dir, "srl.spacemouse", "settings.json")

    def _load_user_settings(self, path: str) -> dict:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
        except Exception as exc:
            carb.log_warn(f"Failed to read settings file {path}: {exc}")
        return {}

    def _save_user_settings(self, path: str, data: dict):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
        except Exception as exc:
            carb.log_warn(f"Failed to write settings file {path}: {exc}")

    def _reset_settings_to_defaults(self):
        defaults = dict(DEFAULT_SETTINGS)
        defaults["device_name"] = DEVICE_NAMES[0] if DEVICE_NAMES else None
        self._settings_data = defaults
        self._save_user_settings(self._settings_path, self._settings_data)
        clear_camera_settings()

        cb_model, dropdown_model = self._models["Engage"]
        self._suppress_setting_write = True
        cb_model.set_value(False)
        dropdown_model.model.get_item_value_model().set_value(0)
        self._on_engage_event(cb_model)

        modes = self._models["Modes"]
        modes[0].set_value(DEFAULT_SETTINGS["mode_translation"])
        modes[1].set_value(DEFAULT_SETTINGS["mode_rotation"])

        self._models["Smoothing Factor"][0].set_value(DEFAULT_SETTINGS["smoothing_factor"])
        self._models["Softmax Temperature"][0].set_value(DEFAULT_SETTINGS["softmax_temp"])
        self._models["Translation Sensitivity"][0].set_value(DEFAULT_SETTINGS["translation_sensitivity"])
        self._models["Rotation Sensitivity"][0].set_value(DEFAULT_SETTINGS["rotation_sensitivity"])
        self._models["Translation Deadband"][0].set_value(DEFAULT_SETTINGS["translation_deadband"])
        self._models["Rotation Deadband"][0].set_value(DEFAULT_SETTINGS["rotation_deadband"])
        self._suppress_setting_write = False

    def _open_reset_confirm(self):
        carb.log_info("[srl.spacemouse] Open reset confirmation")
        if self._reset_confirm_window is not None:
            self._reset_confirm_window.visible = True
            return
        self._reset_confirm_window = ui.Window(
            "Confirm Reset",
            width=360,
            height=140,
            visible=True,
        )
        with self._reset_confirm_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Reset all SpaceMouse settings to defaults?")
                with ui.HStack(spacing=10):
                    ui.Button("Cancel", clicked_fn=self._close_reset_confirm)
                    ui.Button("Reset", clicked_fn=self._confirm_reset)

    def _close_reset_confirm(self):
        if self._reset_confirm_window is not None:
            self._reset_confirm_window.visible = False

    def _confirm_reset(self):
        self._reset_settings_to_defaults()
        self._close_reset_confirm()

    def _run_spnavcam_script(self, quiet: bool = False):
        script_path = SPNAVCAM_SCRIPT_PATH
        if not os.path.isfile(script_path):
            carb.log_error(f"Script not found: {script_path}")
            return False
        try:
            with open(script_path, "r", encoding="utf-8") as handle:
                code = handle.read()
            if self._script_globals is None:
                self._script_globals = {"__file__": script_path, "__name__": "__main__"}
            exec(compile(code, script_path, "exec"), self._script_globals)
            return True
        except Exception as exc:
            msg = str(exc)
            if quiet and (
                "Stage is not ready yet" in msg
                or "No active viewport found" in msg
                or "Active camera is not ready yet" in msg
                or "Active camera prim not found yet" in msg
            ):
                return False
            carb.log_error(f"Failed to run {script_path}: {exc}")
            return False

    def on_shutdown(self):
        self.engage_sub_handle.unsubscribe()
        self.engage_sub_handle = None
        if self._device:
            self._device.close()
            self._device = None
        self._extra_frames = []

        if self._menu_items is not None:
            self._window_cleanup()
        global instance
        instance = None
        return

    def _window_cleanup(self):
        remove_menu_items(self._menu_items, "SRL")
        self._window = None
        self._menu_items = None
        self._models = None
        return

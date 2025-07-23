import threading
import time

import numpy as np
import pygame

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class XboxController(Device):
    """
    Driver for an Xbox-style controller using pygame.

    Axes mapping (may vary by platform):
      0: LS horizontal (→ X)
      1: LS vertical   (→ Y)
      2: LT            (→ Z down)
      3: RS horizontal (→ roll)
      4: RS vertical   (→ pitch)
      5: RT            (→ Z up)
      6: DPAD horizontal (unused)
      7: DPAD vertical   (unused)

    Button mapping:
      0 (A): toggle grasp
      1 (B): reset
      others: ignored
    """

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise IOError("No joystick detected. Connect an Xbox controller.")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        print(f"Detected controller: {self.joy.get_name()}")

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.deadzone = 0.8

        # internal state
        self._axes = [0.0] * self.joy.get_numaxes()
        self._buttons = [0] * self.joy.get_numbuttons()
        self.grasp = False
        self._reset_state = 0
        self._enabled = False

        # orientation state
        self.rotation = np.eye(3)
        self.raw_drotation = np.zeros(3)

        self._display_controls()

        # start polling thread
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _display_controls():
        print("\nXbox controller mapping:")
        print(" Left stick   → move XY")
        print(" Triggers     → move Z")
        print(" Right stick  → roll/pitch")
        print(" LB / RB      → yaw")
        print(" A button     → toggle gripper")
        print(" B button     → reset")
        print(" ESC / back   → quit\n")

    def start_control(self):
        """Enable reading inputs."""
        self._reset_state = 0
        self.grasp = False
        self._enabled = True

    @property
    def control(self):
        """
        Returns a 6-vector: [dx, dy, dz, roll, pitch, yaw]
        scaled by sensitivities.
        """
        # XY from LS
        dx =  self._axes[1] * self.pos_sensitivity
        dy =  self._axes[0] * self.pos_sensitivity
        # Z from triggers: RT (axis 5) minus LT (axis 2)
        dz = (self._axes[5] - self._axes[2]) * 0.5 * self.pos_sensitivity

        # rotation: RS
        roll_input  =  self._axes[3] * self.rot_sensitivity
        pitch_input =  self._axes[4] * self.rot_sensitivity
        # yaw from bumpers
        yaw_input   = (self._buttons[4] - self._buttons[5]) * 0.5 * self.rot_sensitivity

        return np.array([dx, dy, dz, roll_input, pitch_input, yaw_input])

    @property
    def control_gripper(self):
        return 1.0 if self.grasp else 0.0

    def get_controller_state(self):
        """
        Returns:
            dict with keys
              dpos          : 3-vector
              rotation      : 3×3 matrix
              raw_drotation : 3-vector [roll, pitch, yaw]
              grasp         : 0/1
              reset         : 0/1
        """
        ctrl = self.control
        dpos = ctrl[:3]
        roll, pitch, yaw = ctrl[3:]

        # build incremental rotations
        R1 = rotation_matrix(angle=roll,  direction=[1,0,0])[:3,:3]
        R2 = rotation_matrix(angle=pitch,direction=[0,1,0])[:3,:3]
        R3 = rotation_matrix(angle=yaw,  direction=[0,0,1])[:3,:3]
        self.rotation = self.rotation.dot(R1.dot(R2.dot(R3)))
        self.raw_drotation = np.array([roll, pitch, yaw])

        return {
            "dpos": dpos,
            "rotation": self.rotation,
            "raw_drotation": self.raw_drotation.copy(),
            "grasp": float(self.control_gripper),
            "reset": self._reset_state,
        }

    def _poll_loop(self):
        """Background thread: continuously sample axes & handle buttons."""
        while True:
            # Step 1: pump pygame so axis state is fresh
            pygame.event.pump()

            # Step 2: read every axis (always up-to-date)
            for i in range(self.joy.get_numaxes()):
                val = self.joy.get_axis(i)
                # apply deadzone
                self._axes[i] = val if abs(val) > self.deadzone else 0.0

            # Step 3: still handle button presses/releases via events
            for evt in pygame.event.get():
                if evt.type == pygame.JOYBUTTONDOWN:
                    self._buttons[evt.button] = 1
                    if evt.button == 0:   # A
                        self.grasp = not self.grasp
                    elif evt.button == 1: # B
                        self._reset_state = 1
                        self._enabled = False

                elif evt.type == pygame.JOYBUTTONUP:
                    self._buttons[evt.button] = 0

                elif evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

            time.sleep(0.01)


if __name__ == "__main__":
    xb = XboxController()
    xb.start_control()
    while True:
        state = xb.get_controller_state()
        print(state["dpos"], state["raw_drotation"], state["grasp"], state["reset"])
        time.sleep(0.02)

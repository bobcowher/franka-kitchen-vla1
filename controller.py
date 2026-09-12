import time
import numpy as np
import pygame

# FrankaKitchen-v1 action dims, from the MuJoCo actuator list:
#   0-6  panda0_joint1 .. panda0_joint7
#   7    r_gripper_finger_joint
#   8    l_gripper_finger_joint
# Zero holds position, so releasing everything is a no-op rather than a snap home.
#
# Button numbers below were measured on an 8BitDo pad in Switch mode, reporting
# as "Pro Controller" (14 buttons, 4 axes, 1 hat). Changing the pad's input mode
# renumbers the buttons, so these would need re-measuring.
#
# Sticks report up as negative and right as positive on both sticks.

ABORT_BUTTON = 4  # Star
QUIT_BUTTON = 9   # Minus


class Controller:
    def __init__(self):
        self.reset()

        pygame.init()
        pygame.joystick.init()

        # Assuming only one joystick is connected
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def reset(self):
        """Line the controller up with a freshly reset env.

        gripper_closed is sticky, and this object outlives an episode, so
        without this a demo that ended holding something would start the next
        one commanding a close -- against fingers that env.reset() just opened.
        Open rather than None: None left dims 7/8 at zero until the first B or
        X, so identical frames got different action targets depending on
        whether the gripper had been touched yet.
        """
        self.gripper_closed = False

    def get_action(self):
        """
        Map controller input to the robot's action space.

        Returns None when nothing is being pressed, which tells the caller to
        skip the env step rather than advance the sim with a zero action.
        """
        action = np.zeros(9)

        gripper_button_pressed = False

        # Left stick -> joint1 and joint2
        action[0] = self.joystick.get_axis(0) * -1  # left stick horizontal
        action[1] = self.joystick.get_axis(1) * -1  # left stick vertical

        # Right stick -> joint3 and joint4
        action[2] = self.joystick.get_axis(3)       # right stick vertical
        action[3] = self.joystick.get_axis(2) * -1  # right stick horizontal

        if self.joystick.get_button(0):    # B
            self.gripper_closed = True     # close gripper
            gripper_button_pressed = True
            print("Button 0 pressed")
        elif self.joystick.get_button(1):  # A
            action[4] = -1                 # joint5 -
            print("Button 1 pressed")
        elif self.joystick.get_button(2):  # X
            self.gripper_closed = False    # open gripper
            gripper_button_pressed = True
            print("Button 2 pressed")
        elif self.joystick.get_button(3):  # Y
            action[4] = 1                  # joint5 +
            print("Button 3 pressed")
        elif self.joystick.get_button(5):  # L
            action[5] = 1                  # joint6 +  wrist side-to-side tilt
            print("Button 5 pressed")
        elif self.joystick.get_button(6):  # R
            action[5] = -1                 # joint6 -  wrist side-to-side tilt
            print("Button 6 pressed")
        elif self.joystick.get_button(7):  # L2
            action[6] = -1                 # joint7 -
            print("Button 7 pressed")
        elif self.joystick.get_button(8):  # R2
            action[6] = 1                  # joint7 +
            print("Button 8 pressed")

        mask = np.abs(action) >= 0.1
        action = action * mask
        action = np.where(action == -0.0, 0.0, action)

        if np.all(action == 0) and gripper_button_pressed == False:
            action = None
        else:
            if self.gripper_closed == True:
                action[7] = -1.0  # Close gripper
                action[8] = -1.0
            elif self.gripper_closed == False:
                action[7] = 1.0   # Open gripper
                action[8] = 1.0

        return action

    def abort_pressed(self):
        """Star cancels the episode. Not used by get_action, so it stays free."""
        return bool(self.joystick.get_button(ABORT_BUTTON))

    def quit_pressed(self):
        """Minus ends the session. Not used by get_action, so it stays free."""
        return bool(self.joystick.get_button(QUIT_BUTTON))

    def wait_for_release(self):
        """Block until nothing is held, so the next episode starts clean."""
        while any(self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())):
            pygame.event.pump()
            time.sleep(0.05)

    def keep_demo(self):
        """Ask whether to save the episode just finished. B keeps, X discards."""
        print("Demo complete. B to keep, X to discard.")
        self.wait_for_release()
        while True:
            pygame.event.pump()
            if self.joystick.get_button(0):    # B
                self.wait_for_release()
                return True
            elif self.joystick.get_button(2):  # X
                self.wait_for_release()
                return False
            time.sleep(0.05)

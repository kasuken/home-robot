
from time import sleep
import time
from .picarx import Picarx
from robot_hat.music import Music
import threading
import queue
import random
from enum import StrEnum

def forward(car):
    car.forward(5)
    sleep(1)
    car.stop()

def backward(car):
    car.backward(5)
    sleep(1)
    car.stop()

def wave_hands(car):
    car.reset()
    car.set_cam_tilt_angle(20)
    for _ in range(2):
        car.set_dir_servo_angle(-25)
        sleep(.1)
        # car.set_dir_servo_angle(0)
        # sleep(.1)
        car.set_dir_servo_angle(25)
        sleep(.1)
    car.set_dir_servo_angle(0)

def resist(car):
    car.reset()
    car.set_cam_tilt_angle(10)
    for _ in range(3):
        car.set_dir_servo_angle(-15)
        car.set_cam_pan_angle(15)
        sleep(.1)
        car.set_dir_servo_angle(15)
        car.set_cam_pan_angle(-15)
        sleep(.1)
    car.stop()
    car.set_dir_servo_angle(0)
    car.set_cam_pan_angle(0)

def act_cute(car):
    car.reset()
    car.set_cam_tilt_angle(-20)
    for i in range(15):
        car.forward(5)
        sleep(0.02)
        car.backward(5)
        sleep(0.02)
    car.set_cam_tilt_angle(0)
    car.stop()

def rub_hands(car):
    car.reset()
    for i in range(5):
        car.set_dir_servo_angle(-6)
        sleep(.5)
        car.set_dir_servo_angle(6)
        sleep(.5)
    car.reset()

def think(car):
    car.reset()

    for i in range(11):
        car.set_cam_pan_angle(i*3)
        car.set_cam_tilt_angle(-i*2)
        car.set_dir_servo_angle(i*2)
        sleep(.05)
    sleep(1)
    car.set_cam_pan_angle(15)
    car.set_cam_tilt_angle(-10)
    car.set_dir_servo_angle(10)
    sleep(.1)
    car.reset()

def keep_think(car):
    car.reset()
    for i in range(11):
        car.set_cam_pan_angle(i*3)
        car.set_cam_tilt_angle(-i*2)
        car.set_dir_servo_angle(i*2)
        sleep(.05)

def shake_head(car):
    car.stop()
    car.set_cam_pan_angle(0)
    car.set_cam_pan_angle(60)
    sleep(.2)
    car.set_cam_pan_angle(-50)
    sleep(.1)
    car.set_cam_pan_angle(40)
    sleep(.1)
    car.set_cam_pan_angle(-30)
    sleep(.1)
    car.set_cam_pan_angle(20)
    sleep(.1)
    car.set_cam_pan_angle(-10)
    sleep(.1)
    car.set_cam_pan_angle(10)
    sleep(.1)
    car.set_cam_pan_angle(-5)
    sleep(.1)
    car.set_cam_pan_angle(0)

def nod(car):
    car.reset()
    car.set_cam_tilt_angle(0)
    car.set_cam_tilt_angle(5)
    sleep(.1)
    car.set_cam_tilt_angle(-30)
    sleep(.1)
    car.set_cam_tilt_angle(5)
    sleep(.1)
    car.set_cam_tilt_angle(-30)
    sleep(.1)
    car.set_cam_tilt_angle(0)


def depressed(car):
    car.reset()
    car.set_cam_tilt_angle(0)
    car.set_cam_tilt_angle(20)
    sleep(.22)
    car.set_cam_tilt_angle(-22)
    sleep(.1)
    car.set_cam_tilt_angle(10)
    sleep(.1)
    car.set_cam_tilt_angle(-22)
    sleep(.1)
    car.set_cam_tilt_angle(0)
    sleep(.1)
    car.set_cam_tilt_angle(-22)
    sleep(.1)
    car.set_cam_tilt_angle(-10)
    sleep(.1)
    car.set_cam_tilt_angle(-22)
    sleep(.1)
    car.set_cam_tilt_angle(-15)
    sleep(.1)
    car.set_cam_tilt_angle(-22)
    sleep(.1)
    car.set_cam_tilt_angle(-19)
    sleep(.1)
    car.set_cam_tilt_angle(-22)
    sleep(.1)

    sleep(1.5)
    car.reset()

def twist_body(car):
    car.reset()
    for i in range(3):
        car.set_motor_speed(1, 20)
        car.set_motor_speed(2, 20)
        car.set_cam_pan_angle(-20)
        car.set_dir_servo_angle(-10)
        sleep(.1)
        car.set_motor_speed(1, 0)
        car.set_motor_speed(2, 0)
        car.set_cam_pan_angle(0)
        car.set_dir_servo_angle(0)
        sleep(.1)
        car.set_motor_speed(1, -20)
        car.set_motor_speed(2, -20)
        car.set_cam_pan_angle(20)
        car.set_dir_servo_angle(10)
        sleep(.1)
        car.set_motor_speed(1, 0)
        car.set_motor_speed(2, 0)
        car.set_cam_pan_angle(0)
        car.set_dir_servo_angle(0)

        sleep(.1)


def celebrate(car):
    car.reset()
    car.set_cam_tilt_angle(20)

    car.set_dir_servo_angle(30)
    car.set_cam_pan_angle(60)
    sleep(.3)
    car.set_dir_servo_angle(10)
    car.set_cam_pan_angle(30)
    sleep(.1)
    car.set_dir_servo_angle(30)
    car.set_cam_pan_angle(60)
    sleep(.3)
    car.set_dir_servo_angle(0)
    car.set_cam_pan_angle(0)
    sleep(.2)

    car.set_dir_servo_angle(-30)
    car.set_cam_pan_angle(-60)
    sleep(.3)
    car.set_dir_servo_angle(-10)
    car.set_cam_pan_angle(-30)
    sleep(.1)
    car.set_dir_servo_angle(-30)
    car.set_cam_pan_angle(-60)
    sleep(.3)
    car.set_dir_servo_angle(0)
    car.set_cam_pan_angle(0)
    sleep(.2)

def honking(music):
    music.sound_play_threading("../sounds/car-double-horn.wav", 100)

def start_engine(music):
    music.sound_play_threading("../sounds/car-start-engine.wav", 50)

actions_dict = {
    "shake head":shake_head, 
    "nod": nod,
    "wave hands": wave_hands,
    "resist": resist,
    "act cute": act_cute,
    "rub hands": rub_hands,
    "think": think,
    "twist body": twist_body,
    "celebrate": celebrate,
    "depressed": depressed,
    "forward": forward,
    "backward": backward,
}

sounds_dict = {
    "honking": honking,
    "start engine": start_engine,
}


class ActionStatus(StrEnum):
    STANDBY = 'standby'
    THINK = 'think'
    ACTIONS = 'actions'
    ACTIONS_DONE = 'actions_done'

class ActionFlow():
    def __init__(self, car: Picarx) -> None:
        self.car = car
        self.music = Music()
        self.status = ActionStatus.STANDBY
        self.last_status = None
        self.action_queue = queue.Queue()
        self.running = False
        self.thread = None

    def do_action(self, action: str) -> None:
        if action in actions_dict:
            actions_dict[action](self.car)
        elif action in sounds_dict:
            sounds_dict[action](self.music)

    def action_handler(self) -> None:
        last_action_time = time.time()
        action_interval = 5 # seconds
    
        try:
            while self.running:
                # actions
                # ------------------------------
                if self.status == ActionStatus.STANDBY:
                    self.last_status = ActionStatus.STANDBY
                    if time.time() - last_action_time > action_interval:
                        # TODO: standby actions
                        last_action_time = time.time()
                        action_interval = random.randint(2, 6)
                elif self.status == ActionStatus.THINK:
                    if self.last_status != ActionStatus.THINK:
                        self.last_status = ActionStatus.THINK
                        keep_think(self.car)
                elif self.status == ActionStatus.ACTIONS:
                    if self.last_status != ActionStatus.ACTIONS:
                        self.last_status = ActionStatus.ACTIONS
                    _action = self.action_queue.get()
                    self.do_action(_action)
                    time.sleep(0.5)
                    if self.action_queue.empty():
                        self.status = ActionStatus.STANDBY
                        last_action_time = time.time()

                time.sleep(0.01)
        except Exception as e:
            print(f"action handler error: {e}")

    def add_action(self, *actions):
        for action in actions:
            if action not in actions_dict and action not in sounds_dict:
                print(f"action {action} not found")
                continue
            self.action_queue.put(action)
        self.status = ActionStatus.ACTIONS
    
    def set_status(self, status):
        self.status = status

    def wait_actions_done(self):
        while self.status != ActionStatus.STANDBY:
            time.sleep(0.01)

    def start(self):
        self.running = True
        self.status = ActionStatus.STANDBY
        self.action_queue = queue.Queue()
        self.thread = threading.Thread(name="action_handler", target=self.action_handler)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread != None:
            self.thread.join()

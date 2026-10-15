'''
Cliff detection program for Picar-X:

Pay attention to modify the reference value of the grayscale module 
according to the practical usage scenarios.
Auto calibrate grayscale values:
    Please run ./1.cali_grayscale.py
Manual modification:
    Use the following: 
        px.set_cliff_reference([200, 200, 200])
    The reference value should be close to the middle of the line gray value
    and the background gray value.
'''

from picarx import Picarx
from time import sleep

px = Picarx()
# px = Picarx(grayscale_pins=['A0', 'A1', 'A2'])
# manual modify reference value
px.set_cliff_reference([200, 200, 200])

last_state = "safe"

if __name__ == '__main__':
    try:
        while True:
            gm_val_list = px.get_grayscale_data()
            gm_state = px.get_cliff_status(gm_val_list)
            # print("cliff status is: %s" % gm_state)

            if gm_state is False:
                state = "safe"
                px.stop()
            else:
                state = "danger"
                px.backward(80)
                if last_state == "safe":
                    sleep(0.1)

            last_state = state

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: stop and exit")

    finally:
        px.stop()
        sleep(0.1)

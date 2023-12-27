import pystk


def control(aim_point, current_vel, steer_gain=2, skid_thresh=0.5, target_vel=25):
    import numpy as np
    action = pystk.Action()

    steer_angle = steer_gain * aim_point[0]

    # Compute accelerate
    action.acceleration = 1.0 if current_vel < target_vel else 0.0

    # Compute steering
    action.steer = np.clip(steer_angle * steer_gain, -1, 1)

    # Compute skidding
    if abs(steer_angle) > skid_thresh:
        action.drift = True
    else:
        action.drift = False

    action.nitro = True

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)

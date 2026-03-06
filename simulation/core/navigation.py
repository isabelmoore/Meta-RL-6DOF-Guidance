
def roll_circle_clip(roll_delta):
    if roll_delta > 180.0:
        roll_delta -= 360.0
    elif roll_delta <= -180.0:
        roll_delta += 360.0
    return roll_delta

def delta_heading(target, current):
    diff = (target - current + 180) % 360 - 180
    return diff


if __name__ == "__main__":
    for test_phi_ref, test_phi in [(10,90), (10,170), (-45,170), (-100, -45), (0,180), (180,0)]:
        print(f"phi_ref: {test_phi_ref}, phi: {test_phi} => roll_delta: {roll_circle_clip(test_phi_ref, test_phi)}")

    for current, target in [(10,90), (350,10), (45,170), (170,45), (0,180), (180,0)]:
        print(f"current: {current}, target: {target} => heading_delta: {delta_heading(current, target)}")
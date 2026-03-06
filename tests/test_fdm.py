# Copyright (c) 2026 Isabel Moore. All rights reserved.
import jsbsim
import os

print("Initializing JSBSim...")
fdm = jsbsim.FGFDMExec('.', None)
script_path = 'jsbsim_data/scripts/AIM_test.xml'
print(f"Loading script: {script_path}")

if not os.path.exists(script_path):
    print(f"Script file not found at {os.path.abspath(script_path)}")
    exit(1)

res = fdm.load_script(script_path)
print(f"Load script result: {res}")

if not res:
    print("Failed to load script. Check JSBSim output.")
else:
    print("Script loaded successfully.")
    print("Running IC...")
    try:
        fdm.run_ic()
        print("IC run complete.")
    except Exception as e:
        print(f"Error running IC: {e}")

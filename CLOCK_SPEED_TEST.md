# AirSim clock-speed check

The original 20-step environment benchmark measured AirSim RPC, movement, collision checks, and depth-image capture. These measurements were short environment checks, not full PPO training results.

| Clock setting | Short-check result |
| --- | --- |
| 1x | 20/20 steps completed; about 1x simulated time |
| 2x | 20/20 steps completed; about 2x simulated time in the short segment |
| 3x-5x | Short runs ended early after the drone fell below the floor |
| 10x | Invalid position/collision behavior occurred after an initial fast timestamp advance |
| 20x | The process ran, but the AirSim simulation timestamp did not advance; unusable for training |
| 30x | Not validated as a stable training setting |

The project's target of 20-30x real time has not been demonstrated. Training was run at 1x for stability. Do not interpret the configured `ClockSpeed` number as measured training throughput. The benchmark script can be run after placing the Easy environment at `environment/Easy`:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\benchmark_clock_speed.ps1 -Speeds "1,2" -Steps 20
```

AirSim's documentation notes that high clock speeds can degrade simulation fidelity: https://microsoft.github.io/AirSimExtensions/settings/#clockspeed

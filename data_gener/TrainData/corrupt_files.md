Corrupt files in the nyuv2 dataset that we may need to re-download


### Imread Problem Files

Scene: `home_office_0002`
* `0010`
  * Error Message: `ERROR: MATLAB:imagesci:pnmgeti:corruptFile Garbage found where image data was expected`
  * The error happend in the `imgRgb = imread...` line trying to read `'r-1315165247.616626-1005556086.ppm'`
  * Although, MATLAB can't read it, my Image Viewer software can, and I can read it with skimage.
  * Could be fixed by reading the file, saving as PNG, and then saving back to PPM to replace it...

Scene: `living_room_0026`
* `0008` -> Same error as `home_office_0002_0010`

Scene: `living_room_0037`
* `0014` -> Same error as `home_office_0002_0010`

Scene: `living_room_0060`
* `0027` -> Same error as `home_office_0002_0010`

Scene: `living_room_0022`
* `0123` -> Same error as `home_office_0002_0010`

Scene: `living_room_0047a`
* `0050` -> Same error as `home_office_0002_0010`

Scene: `office_0001a`
* `0020` -> Same error as `home_office_0002_0010`

### MATLAB sizeDimensionsMustMatch

Scene: `living_room_0001b`
* `0003`
  * Crash on `ConvertRGBD.m` when calling `fill_depth_cross_bf`
  * Inside `fill_depth_cross_bf` the crash happens in the line `imgDepth = imgDepthAbs ./ maxDepthObs;`
  * The problem is that this file contains 0 valid pixels. So the `maxDepthObs` variable has 0 dimensions.

### Empty Scenes

**NOTE:** I re-downloaded the files and they were still empty...

Scene: `living_room_0023a`
Scene: `living_room_0023b`
Scene: `living_room_0024a`
Scene: `living_room_0024b`
Scene: `living_room_0024c`


### SegFault Files

Scene: `living_room_0059`
* `0195` --> Seg Faults
  * Seg Fault happens in line: `[albedo, ~] = intrinsic_decomp(I, S, imgDepthFilled, 0.0001, 0.8, 0.5);`
  * Inside `intrinsic_decomp` seg fault happens in `nNeighbors = getGridLLEMatrix(nMap_p, vMap_p, 50, 6, 12);`
* `0204` --> Seg Faults
* `0208` --> Seg Faults

Scene: `living_room_0057`
* `0123` --> Seg Faults

Scene: `office_0001c`
* `0083` --> Seg Faults

Scene: `dining_room_0010`
* `0027` --> Seg Faults

Scene: `living_room_0040`
* `0032` --> Seg Faults
* `0061` --> Seg Faults
* `0087` --> Seg Faults

Scene: `living_room_0073`
* `0060` --> Seg Faults

Scene: `dining_room_0020`
* `0120` --> Seg Faults

Scene: `dining_room_0022`
* `0030` --> Seg Faults

Scene: `dining_room_0024`
* `0025` --> Seg Faults
* `0029` --> Seg Faults

Scene: `study_room_0003`
* `0001`, `0007`, `0008`, `0010`, `0024`, `0029`, `0061`, `0085`, `0097`, `0101` 
  * Seg Faults

Scene: `study_room_005b`
* `0045`

Scene: `study_room_0007`
* `0017`


----

Crash on scene `living_room_0059_0195`
* Segmentation violation:
  Crash Mode               : continue (default)
  Default Encoding         : UTF-8
  Deployed                 : false
  Desktop Environment      : ubuntu:GNOME
  GNU C Library            : 2.31 stable
  Graphics Driver          : Unknown hardware 
  Graphics card 1          : 0x10de ( 0x10de ) 0x1e84 Version 0.0.0.0 (0-0-0)
  Graphics card 2          : 0x8086 ( 0x8086 ) 0x9bc5 Version 0.0.0.0 (0-0-0)
  Java Version             : Java 1.8.0_202-b08 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode
  MATLAB Architecture      : glnxa64
  MATLAB Entitlement ID    : 1998947
  MATLAB Root              : /usr/local/MATLAB/R2021b
  MATLAB Version           : 9.11.0.1873467 (R2021b) Update 3
  OpenGL                   : hardware
  Operating System         : Ubuntu 20.04.4 LTS
  Process ID               : 149295
  Processor ID             : x86 Family 6 Model 165 Stepping 5, GenuineIntel
  Session Key              : 48328db8-df0c-4ca5-a167-bb3c95a16de0
  Static TLS mitigation    : Enabled: Full
  Window System            : The X.Org Foundation (12013000), display :0

Fault Count: 1


Abnormal termination:
Segmentation violation

Current Thread: 'MCR 0 interpret' id 140214418523904

Register State (from fault):
  RAX = ffffffffc0000000  RBX = 0000000000000006
  RCX = 00007f844a1503d0  RDX = 0000000000000018
  RSP = 00007f86369b5be0  RBP = 0000000000000006
  RSI = 00007f844a1577d0  RDI = 00007f844a153e00

   R8 = 00007f8464ff6690   R9 = 00007f86369b5c20
  R10 = 00007f8473936650  R11 = 0000000000000028
  R12 = 00007f844a1577d0  R13 = 00007f844a153e00
  R14 = 0000000000000018  R15 = 00007f86369b5f60

  RIP = 00007f8473936776  EFL = 0000000000010246

   CS = 0033   FS = 0000   GS = 0000

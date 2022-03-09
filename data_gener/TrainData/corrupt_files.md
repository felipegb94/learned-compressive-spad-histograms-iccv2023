Corrupt files in the nyuv2 dataset that we may need to re-download


Scene: home_office_0002
* 0010

Scene: living_room_0026
* 0008

Scene: living_room_0059
* 0195 --> Seg Fauls


----

Crash on scene Living_room_0059_0195
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

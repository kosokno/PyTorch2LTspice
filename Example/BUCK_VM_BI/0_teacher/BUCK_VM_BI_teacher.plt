[Frequency Response Analysis]
{
   Npanes: 1
   {
      traces: 3 {2,0,"gain_1"} {3,0,"1/probe_1comp"} {4,0,"probe_1plant"}
      X: ('K',0,10,0,500000)
      Y[0]: (' ',0,0.001,60,1e+33)
      Y[1]: (' ',0,-300,60,300)
      Log: 1 2 0
      LargePixels: 1
      PltMag: 1
      PltPhi: 1 0
   }
}
[Transient Analysis]
{
   Npanes: 2
   Active Pane: 1
   {
      traces: 2 {524290,0,"V(out)"} {524291,0,"V(ref)"}
      X: ('m',0,0,0.004,0.04052)
      Y[0]: (' ',0,0,5,55)
      Y[1]: ('_',0,1e+308,0,-1e+308)
      Volts: (' ',0,0,0,0,5,55)
      Log: 0 0 0
   },
   {
      traces: 1 {268959748,0,"V(nnpwm)"}
      X: ('m',0,0,0.004,0.04052)
      Y[0]: (' ',1,0,0.1,1)
      Y[1]: ('_',0,1e+308,0,-1e+308)
      Volts: (' ',0,0,0,0,0.1,1)
      Log: 0 0 0
   }
}

import rtde_control, rtde_receive, time

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.130")
rtde_c = rtde_control.RTDEControlInterface("192.168.1.130")

rtde_c.zeroFtSensor()
time.sleep(0.2)
while True:
    print(rtde_r.getActualTCPForce())
    time.sleep(0.2)
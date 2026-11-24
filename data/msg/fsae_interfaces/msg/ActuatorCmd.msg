# Raw actuator command for the bench-rig hub Arduino (open-loop efforts).
# Assembled upstream (Stanley temporarily; a longitudinal controller + teleop mux
# later) and serialised by fsae_serial_bridge as a newline-terminated line of the
# form: S{steering} T{throttle} B{brake}
# The Arduino executes these directly - it does NOT interpret speed.
# NB: keep this comment free of backslashes - rosidl copies .msg comments verbatim
# into the generated IDL string, and an unescaped backslash breaks IDL generation.
std_msgs/Header header
float32 steering_angle    # degrees (target steering angle)
float32 throttle_percent  # 0-100
float32 brake_percent     # 0-100
bool armed                # Deadman held. false = the serial bridge sends a neutral frame
                          # (S0.0 T0.0 B0.0) whatever the fields above say.

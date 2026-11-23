# Raspberry Pi 3B+ + SIM7600EI + Jio + Mac live dashboard

This version is **continuous**: there is no 5-cycle limit and no 8-second experiment limit. The Pi samples BMP280 and SCD4X forever at the configured interval (default 1 second), publishes every sample over MQTT, and continues until you press `Ctrl+C` or stop the service. The Mac dashboard displays live values and the Mac app appends every received sample to a persistent JSON Lines history file.

It also includes:
- BMP280 pressure and temperature (I2C address `0x76`)
- SCD4X CO2, temperature and relative humidity (latest valid SCD4X reading is retained between sensor updates)
- Dashboard-controlled continuous microphone recording: ON starts consecutive 30-second WAV recordings; OFF stops after the current file finishes uploading, using ALSA device `plughw:2,0`
- Each 30-second WAV is transferred from Pi to Mac in MQTT chunks, saved in `mac_server/data/audio/`, indexed with its recording timestamp in `audio_index.jsonl`, and available for playback/download in the dashboard
- Dashboard ping command, physical LED flash on Pi GPIO17, and MQTT acknowledgement
- MQTT reconnect handling

## Network design used by this package

This package uses **PPP internet through the SIM7600EI**, then normal Python MQTT. This is the recommended route because it supports both continuous MQTT and audio file transfer. The Pi and Mac each connect outbound to the MQTT broker; the Mac does not need a public IP address.

The SIM7600 UART is owned by PPP while online. Do not run AT-command tests on `/dev/serial0` while PPP is running. Your prior AT tests confirmed SIM ready and network registration, but that alone does not prove that the mobile data session is active. The PPP test below must succeed before MQTT can work.

## 0. Configure the topic root

In BOTH files below, replace `rpi-experiment/change-this-unique-id` with your own long, hard-to-guess topic, and make sure the strings match exactly:
- `raspberry_pi/pi_agent.py`
- `mac_server/app.py`

Example: `rpi-experiment/lab-device-7f91c2-unique-random-text`.

The included `broker.hivemq.com:1883` is a **public, unauthenticated, unencrypted test broker**. Anyone can potentially read or publish to public topics. Use a private broker with authentication and TLS before using this for sensitive or production data. A random topic name reduces accidental collisions; it is not security.

## 1. Hardware

### SIM7600EI UART (existing connection)
- SIM7600 TXD -> Pi GPIO15/RXD, physical pin 10
- SIM7600 RXD -> Pi GPIO14/TXD, physical pin 8
- SIM7600 GND -> Pi GND, physical pin 6
- Keep the SIM7600 on its correctly rated external supply with adequate peak current, common ground, and antenna connected. Do not power a bare modem from the Pi 3.3V pin. Verify UART logic-level compatibility for your exact module/carrier.
- SIM APN configured in PPP chat script: `jionet`.

### Signal LED
- Pi GPIO17 (BCM numbering, physical pin 11) -> 220–330 ohm resistor -> LED anode/long leg
- LED cathode/short leg -> Pi GND (physical pin 9)

Keep your working BMP280/SCD4X I2C and microphone wiring unchanged. Check that no other device uses GPIO17.

## 2. Enable Pi serial port for PPP

In `sudo raspi-config`:
- Interface Options -> Serial Port -> login shell over serial: **No**
- Serial hardware enabled: **Yes**

Then run:

```bash
sudo systemctl disable --now serial-getty@ttyS0.service
sudo apt update
sudo apt install ppp chat alsa-utils python3-venv python3-dev -y
```

Your earlier setup showed `/dev/serial0` mapping to `/dev/ttyS0`, and UART AT commands worked with `sudo`. PPP will need exclusive access to this serial port. The Pi 3 mini-UART can be clock-sensitive; if PPP is unstable, configure a stable UART/clock mapping for your Pi and update the serial device in `sim7600-ppp-options`.

## 3. Start cellular PPP on the Pi

Copy this project folder to the Pi at exactly:
`/home/piuser/RPi_SIM7600_Live_Experiment`

For example, from a Mac terminal in the folder containing the extracted project, copy it using your Pi's reachable LAN address:

```bash
scp -r RPi_SIM7600_Live_Experiment piuser@raspberrypi3.local:/home/piuser/
```

If `.local` does not resolve, use the Pi's LAN IP instead. Then on the Pi:

```bash
cd /home/piuser/RPi_SIM7600_Live_Experiment/raspberry_pi
sudo chmod +x ppp-chat-connect
sudo pppd file /home/piuser/RPi_SIM7600_Live_Experiment/raspberry_pi/sim7600-ppp-options
```

Keep this terminal open. In a second Pi terminal, test:

```bash
ip addr show ppp0
ip route
ping -I ppp0 -c 4 1.1.1.1
```

You need a `ppp0` interface and successful ping. If PPP does not connect, stop here and share the complete PPP/chat output; do not proceed assuming the SIM registration means internet is working. Some modem firmware/carriers require a different dial string or flow-control configuration.

## 4. Install and run the continuous Pi agent

In another Pi terminal:

```bash
cd /home/piuser/RPi_SIM7600_Live_Experiment/raspberry_pi
sudo apt install python3-venv python3-dev -y
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Test the microphone before starting:

```bash
arecord -l
arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 2 /tmp/mic-test.wav
```

Start the agent (sudo is used because your earlier GPIO/serial device permissions required it):

```bash
sudo .venv/bin/python pi_agent.py
```

Expected log: MQTT connected, then sensor samples with increasing `sample` numbers. It does not stop after five cycles. Use `Ctrl+C` to stop it. Do not run two copies simultaneously.

If `RPi.GPIO` or Adafruit imports fail inside the virtual environment, install the corresponding packages into the system Python or recreate the venv with `--system-site-packages` as above. `python3-rpi.gpio` can also be installed with `sudo apt install python3-rpi.gpio`.

## 5. Run the Mac dashboard

Before starting the dashboard, create `mac_server/.env` with the Supabase project URL and key:

```text
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key
```

Keep `.env` private and do not commit it. The Supabase `sound_samples` table must have the columns `device_id`, `sample`, `timestamp`, `timestamp_local`, `sensor_data` (JSON/JSONB), and `audio_path`; `id` should be generated by the database. Each received MQTT telemetry sample is inserted into this table. The dashboard also continues appending the same samples to `mac_server/data/sensor_data.jsonl`. Supabase insert errors are printed in the dashboard terminal.

On the Mac, extract the ZIP. Open Terminal in `RPi_SIM7600_Live_Experiment/mac_server`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. Keep the Mac connected to the internet and keep this Flask process running. If the Mac and Pi share a local network, other devices can use `http://MAC-LAN-IP:5000`; the Pi itself does not need to reach the Mac over the cellular connection.

## 6. Dashboard functions

- Live sensor cards refresh every second in the browser.
- Pi samples and sends telemetry every `SENSOR_INTERVAL` seconds (default 1 second).
- `Test signal / LED` sends a ping; the Pi flashes GPIO17 and publishes an acknowledgement.
- The audio control is an ON/OFF toggle. ON starts consecutive 30-second mono 16 kHz WAV recordings from `plughw:2,0`. Each recording is uploaded in MQTT chunks before the next recording begins, so files are sent one by one. OFF prevents the next recording from starting; the current 30-second recording and its upload are allowed to finish.
- The Mac stores every completed WAV in `mac_server/data/audio/`. The recording timestamp, receive timestamp, filename, and file size are appended to `mac_server/data/audio_index.jsonl`. The dashboard lists saved recordings with playback and download links.
- `Download complete sensor history (JSON Lines)` downloads `mac_server/data/sensor_data.jsonl`. Each line is one complete JSON sample. This append-only file is the full local history; the web table only shows recent samples in memory.

## 7. Important reliability notes

- “Continuous all the time” means the Python process must remain running, the Pi must remain powered, PPP/mobile data must remain connected, and the Mac Flask app must remain running to receive and store data. If any of these stop, no remote live updates are possible. Audio recording continues only while the dashboard audio toggle is ON. Paho reconnects MQTT after temporary broker/network drops; the Pi agent does not currently buffer missed samples while internet is unavailable.
- The dashboard can only show data received while it is running. For unattended operation, configure PPP and the Pi agent as systemd services after the manual test succeeds.
- The Mac stores all received samples in JSONL, while the dashboard keeps only the latest 300 samples in RAM for a fast recent table.
- SCD4X has its own measurement cadence, slower than 1 second; the agent retains the latest valid CO2/temperature/humidity values between updates. BMP280 is read each loop.
- Public HiveMQ is for testing only. Public broker delivery is not guaranteed and topics are not private. Use a private authenticated TLS broker for dependable deployment.
- Audio is sent through MQTT as chunks. Very long recordings take longer to transfer and may be affected by mobile coverage/broker limits. Start with 5–10 seconds to test.

## File layout

```text
RPi_SIM7600_Live_Experiment/
├── raspberry_pi/
│   ├── pi_agent.py
│   ├── requirements.txt
│   ├── sim7600-ppp-options
│   └── ppp-chat-connect
└── mac_server/
    ├── app.py
    └── requirements.txt
```

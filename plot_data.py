import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = r"C:\Users\lalas\Downloads\U0201_LeftWrist.csv"
df = pd.read_csv(csv_file_path)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['accel_x'], label='Accel X ')
plt.plot(df['timestamp'], df['accel_y'], label='Accel Y ')
plt.plot(df['timestamp'], df['accel_z'], label='Accel Z ')
plt.title('Accelerometer Data')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['gyro_x'], label='Gyro X ')
plt.plot(df['timestamp'], df['gyro_y'], label='Gyro Y ')
plt.plot(df['timestamp'], df['gyro_z'], label='Gyro Z ')
plt.title('Gyroscope Data')
plt.xlabel('Timestamp')
plt.ylabel('Gyro')
plt.legend()

plt.tight_layout()
plt.show()

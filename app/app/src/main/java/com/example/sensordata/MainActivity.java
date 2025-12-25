package com.example.sensordata;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.PrintWriter;
import java.net.Socket;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    // --- KONFIGURACJA SIECI ---
    // Pamiętaj o komendzie w terminalu komputera: adb reverse tcp:5555 tcp:5555
    private static final String SERVER_IP = "127.0.0.1";
    private static final int SERVER_PORT = 5555;

    private TextView tvStatus, tvAccData, tvGyroData;
    private Button btnConnect;
    private SensorManager sensorManager;
    private Sensor linearAcceleration, gyroscope; // Zmieniona nazwa zmiennej dla jasności

    // Surowe dane do wysyłki
    private volatile float[] rawAcc = new float[]{0, 0, 0};
    private volatile float[] rawGyro = new float[]{0, 0, 0};

    // Zmienne do obsługi UI
    private long lastUiUpdate = 0;

    private Socket socket;
    private PrintWriter outputWriter;
    private volatile boolean isConnected = false;
    private Thread networkThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvStatus);
        tvAccData = findViewById(R.id.tvAccData);
        tvGyroData = findViewById(R.id.tvGyroData);
        btnConnect = findViewById(R.id.btnConnect);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            // --- KLUCZOWA ZMIANA ---
            // Używamy TYPE_LINEAR_ACCELERATION, aby usunąć grawitację
            linearAcceleration = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        }

        btnConnect.setOnClickListener(v -> {
            if (!isConnected) startConnection();
            else stopConnection();
        });
    }

    private void startConnection() {
        isConnected = true;
        tvStatus.setText("Connecting...");
        tvStatus.setTextColor(0xFFFFFF00); // Żółty

        networkThread = new Thread(() -> {
            try {
                socket = new Socket(SERVER_IP, SERVER_PORT);
                outputWriter = new PrintWriter(socket.getOutputStream(), true);

                runOnUiThread(() -> {
                    tvStatus.setText("CONNECTED");
                    tvStatus.setTextColor(0xFF00FF00); // Zielony
                    btnConnect.setText("DISCONNECT");
                    btnConnect.setBackgroundColor(0xFFFF0000);
                });

                while (isConnected) {
                    if (outputWriter != null) {
                        // Pobranie danych
                        float ax = rawAcc[0];
                        float ay = rawAcc[1];
                        float az = rawAcc[2];
                        float gx = rawGyro[0];
                        float gy = rawGyro[1];
                        float gz = rawGyro[2];

                        // Formatowanie JSON
                        String json = String.format(Locale.US,
                                "{\"acc_x\": %.4f, \"acc_y\": %.4f, \"acc_z\": %.4f, " +
                                        "\"gyro_x\": %.4f, \"gyro_y\": %.4f, \"gyro_z\": %.4f}",
                                ax, ay, az, gx, gy, gz);

                        outputWriter.println(json);

                        // 10ms = 100Hz (wysoka precyzja dla fizyki)
                        Thread.sleep(10);
                    }
                }
            } catch (Exception e) {
                Log.e("SensorApp", "Network error", e);
                stopConnection();
            }
        });
        networkThread.start();
    }

    private void stopConnection() {
        isConnected = false;
        try {
            if (socket != null) socket.close();
            if (networkThread != null) networkThread.interrupt();
        } catch (Exception e) { e.printStackTrace(); }

        runOnUiThread(() -> {
            tvStatus.setText("DISCONNECTED"); tvStatus.setTextColor(0xFFFF0000); // Czerwony
            btnConnect.setText("CONNECT");
            btnConnect.setBackgroundColor(0xFF00AA00);
        });
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        // --- ZMIANA: Sprawdzamy TYPE_LINEAR_ACCELERATION ---
        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            rawAcc = event.values.clone();
        }
        else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            rawGyro = event.values.clone();
        }

        // Aktualizacja ekranu telefonu (co 100ms, żeby nie zamulać)
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastUiUpdate > 100) {
            lastUiUpdate = currentTime;

            // Dodaję info "LinAcc" żebyś wiedział, że to ten tryb
            String accText = String.format(Locale.US, "LinAcc X: %.2f\nLinAcc Y: %.2f\nLinAcc Z: %.2f",
                    rawAcc[0], rawAcc[1], rawAcc[2]);
            String gyroText = String.format(Locale.US, "Gyro X: %.2f\nGyro Y: %.2f\nGyro Z: %.2f",
                    rawGyro[0], rawGyro[1], rawGyro[2]);

            tvAccData.setText(accText);
            tvGyroData.setText(gyroText);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override
    protected void onResume() {
        super.onResume();
        if (sensorManager != null) {
            // SENSOR_DELAY_GAME (ok. 50Hz) to dobry kompromis
            if (linearAcceleration != null)
                sensorManager.registerListener(this, linearAcceleration, SensorManager.SENSOR_DELAY_GAME);
            if (gyroscope != null)
                sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(this);
        stopConnection();
    }
}
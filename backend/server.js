/**
 * Node.js Server - AI Du Doan Gia Nha HCM
 * Su dung Express + goi Python predict qua HTTP (predict_server)
 */
const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');

const { preprocessInput } = require('./preprocessing');

const app = express();
const PORT = process.env.PORT || 8000;
const PREDICT_PORT = process.env.PREDICT_PORT || 5001;

app.use(cors());
app.use(express.json());

// Serve frontend
app.use(express.static(path.join(__dirname, '..', 'frontend')));

// Spawn predict server khi start
let predictServerProcess = null;
let predictServerReady = false;
let predictRestartCount = 0;
const MAX_PREDICT_RESTARTS = 10;

function waitForPredictServer(retries, interval, callback) {
    if (retries <= 0) return callback(false);
    const req = http.get(`http://127.0.0.1:${PREDICT_PORT}/health`, { timeout: 2000 }, (res) => {
        let body = '';
        res.on('data', (c) => { body += c; });
        res.on('end', () => {
            try {
                const data = JSON.parse(body);
                if (data.status === 'healthy') return callback(true);
            } catch (e) {}
            setTimeout(() => waitForPredictServer(retries - 1, interval, callback), interval);
        });
    });
    req.on('error', () => {
        setTimeout(() => waitForPredictServer(retries - 1, interval, callback), interval);
    });
    req.on('timeout', () => { req.destroy(); setTimeout(() => waitForPredictServer(retries - 1, interval, callback), interval); });
}

function findPython() {
    const venvPython = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
    const fs = require('fs');
    if (fs.existsSync(venvPython)) return venvPython;
    return 'python';
}

function startPredictServer() {
    const scriptPath = path.join(__dirname, 'predict_server.py');
    const pythonCmd = findPython();
    predictServerProcess = require('child_process').spawn(pythonCmd, [scriptPath, String(PREDICT_PORT)], {
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    });

    predictServerProcess.stdout.on('data', (d) => {
        const msg = d.toString().trim();
        if (msg) console.log(`[predict] ${msg}`);
    });

    predictServerProcess.stderr.on('data', (d) => {
        const msg = d.toString().trim();
        if (msg && !msg.includes('WARNING')) console.log(`[predict] ${msg}`);
    });

    predictServerProcess.on('error', (err) => {
        console.error('Failed to start predict server:', err.message);
    });

    predictServerProcess.on('close', (code) => {
        predictRestartCount++;
        if (predictRestartCount >= MAX_PREDICT_RESTARTS) {
            console.error(`Predict server crashed ${MAX_PREDICT_RESTARTS} times. Giving up.`);
            predictServerProcess = null;
            predictServerReady = false;
            return;
        }
        console.error(`Predict server exited with code ${code}. Restarting (${predictRestartCount}/${MAX_PREDICT_RESTARTS}) in 3s...`);
        predictServerProcess = null;
        predictServerReady = false;
        setTimeout(startPredictServer, 3000);
    });

    console.log(`  Predict server starting on port ${PREDICT_PORT}...`);

    waitForPredictServer(20, 500, (ready) => {
        if (ready) {
            predictServerReady = true;
            predictRestartCount = 0;
            console.log(`  Predict server is ready!`);
        } else {
            console.error(`  Predict server failed to start after 10s`);
        }
    });
}

function callPredictServer(data, callback) {
    const jsonData = JSON.stringify(data);

    const options = {
        hostname: '127.0.0.1',
        port: PREDICT_PORT,
        path: '/predict',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(jsonData),
        },
        timeout: 10000,
    };

    const req = http.request(options, (res) => {
        let body = '';
        res.on('data', (chunk) => { body += chunk; });
        res.on('end', () => {
            try {
                const result = JSON.parse(body);
                callback(null, result);
            } catch (e) {
                callback(new Error('Invalid response from predict server'));
            }
        });
    });

    req.on('error', (err) => callback(err));
    req.on('timeout', () => {
        req.destroy();
        callback(new Error('Predict server timeout'));
    });

    req.write(jsonData);
    req.end();
}

// Meta data
app.get('/meta', (req, res) => {
    const { DISTRICTS, HUONG, PHAP_LY, CHAT_LUONG_XAY_DUNG,
            VIEW_TYPES, WARDS } = require('./preprocessing');

    const DU_AN_CAN_HO = [
        "Vinhomes Grand Park", "Sunrise City", "The Estella", "Masteri Thao Dien",
        "Saigon Pearl", "Vinci Grand Plaza", "The Sun Avenue", "Icon 56",
        "Saigon Royal", "D'Edge", "City Garden", "Midtown", "Charmington",
        "SC VivoCity", "Centre Mall", "Happy Valley", "Millennium"
    ];

    res.json({
        districts: DISTRICTS,
        wards: WARDS,
        huong: HUONG,
        phap_ly: PHAP_LY,
        chat_luong_xay_dung: CHAT_LUONG_XAY_DUNG,
        view: VIEW_TYPES,
        du_an_can_ho: DU_AN_CAN_HO,
    });
});

// Predict endpoint
app.post('/predict', (req, res) => {
    try {
        const { house_type } = req.body;

        if (!house_type || house_type !== 'can_ho') {
            return res.status(400).json({ error: 'Invalid house_type. Only "can_ho" is supported.' });
        }

        if (!predictServerReady) {
            return res.status(503).json({ error: 'Predict server is starting up, please try again in a few seconds' });
        }

        // Preprocess input
        const features = preprocessInput(req.body, house_type);

        // Goi predict server qua HTTP
        callPredictServer({ house_type, features }, (err, result) => {
            if (err) {
                console.error('Predict server error:', err.message);
                return res.status(500).json({ error: 'Prediction failed', detail: err.message });
            }

            if (result.error) {
                return res.status(500).json({ error: result.error });
            }

            res.json(result);
        });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// Health check
app.get('/health', (req, res) => {
    // Kiem tra predict server
    const options = {
        hostname: '127.0.0.1',
        port: PREDICT_PORT,
        path: '/health',
        method: 'GET',
        timeout: 3000,
    };

    const healthReq = http.request(options, (proxyRes) => {
        let body = '';
        proxyRes.on('data', (chunk) => { body += chunk; });
        proxyRes.on('end', () => {
            try {
                const predictHealth = JSON.parse(body);
                res.json({
                    status: 'healthy',
                    version: '3.0.0',
                    predict_server: predictHealth
                });
            } catch (e) {
                res.json({ status: 'degraded', version: '3.0.0', predict_server: 'unavailable' });
            }
        });
    });

    healthReq.on('error', () => {
        res.json({ status: 'degraded', version: '3.0.0', predict_server: 'unavailable' });
    });

    healthReq.on('timeout', () => {
        healthReq.destroy();
        res.json({ status: 'degraded', version: '3.0.0', predict_server: 'timeout' });
    });

    healthReq.end();
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down...');
    if (predictServerProcess) {
        predictServerProcess.kill();
    }
    process.exit(0);
});

process.on('SIGTERM', () => {
    if (predictServerProcess) {
        predictServerProcess.kill();
    }
    process.exit(0);
});

// Start server
app.listen(PORT, () => {
    console.log(`\n  AI Du Doan Gia Nha HCM - Node.js Server v3.0`);
    console.log(`  http://localhost:${PORT}\n`);
    startPredictServer();
});

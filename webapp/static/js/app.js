/**
 * F-YOLO Hybrid PestVision — Frontend Application v2
 * Renders CNN score, YOLO score, and Fuzzy severity independently per detection.
 */

(function () {
    'use strict';

    // ─── DOM Elements ──────────────────────────────────────────────────────────
    const headerStatus    = document.getElementById('headerStatus');
    const pipeCNN         = document.getElementById('pipeCNN');
    const pipeYOLO        = document.getElementById('pipeYOLO');
    const pipeFuzzy       = document.getElementById('pipeFuzzy');
    const uploadZone      = document.getElementById('uploadZone');
    const uploadLoading   = document.getElementById('uploadLoading');
    const lsCNN           = document.getElementById('lsCNN');
    const lsYOLO          = document.getElementById('lsYOLO');
    const lsFuzzy         = document.getElementById('lsFuzzy');
    const fileInput       = document.getElementById('fileInput');
    const heroSection     = document.getElementById('heroSection');
    const uploadSection   = document.getElementById('uploadSection');
    const resultsSection  = document.getElementById('resultsSection');
    const categoriesSection = document.getElementById('categoriesSection');

    const totalDetections = document.getElementById('totalDetections');
    const inferenceTime   = document.getElementById('inferenceTime');
    const imageSize       = document.getElementById('imageSize');
    const btnNewScan      = document.getElementById('btnNewScan');
    const originalImage   = document.getElementById('originalImage');
    const annotatedImage  = document.getElementById('annotatedImage');
    const panelFilename   = document.getElementById('panelFilename');
    const panelCount      = document.getElementById('panelCount');
    const detectionsGrid  = document.getElementById('detectionsGrid');
    const noDetections    = document.getElementById('noDetections');
    const cnnGlobalBanner = document.getElementById('cnnGlobalBanner');
    const cnnGlobalResult = document.getElementById('cnnGlobalResult');
    const cnnGlobalConf   = document.getElementById('cnnGlobalConf');

    const API_BASE     = '';
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

    // ─── Health Check ──────────────────────────────────────────────────────────

    async function checkHealth() {
        try {
            const res  = await fetch(`${API_BASE}/api/health`);
            const data = await res.json();

            if (data.status === 'healthy' && data.model_loaded) {
                if (data.custom_model) {
                    headerStatus.className = 'header-status online';
                    headerStatus.querySelector('.status-text').textContent = 'Hybrid Ready';
                } else {
                    headerStatus.className = 'header-status fallback';
                    headerStatus.querySelector('.status-text').textContent = 'Fallback Model';
                }
            } else {
                headerStatus.className = 'header-status';
                headerStatus.querySelector('.status-text').textContent = 'Model Error';
            }

            // Update pipeline badges
            if (data.cnn_loaded) {
                pipeCNN.classList.add('active');
                pipeCNN.title = 'CNN (MobileNetV2) — Loaded ✅';
            } else {
                pipeCNN.classList.remove('active');
                pipeCNN.classList.add('unavail');
                pipeCNN.title = 'CNN not loaded — train & place cnn_pest_model.h5';
            }

            if (data.model_loaded) {
                pipeYOLO.classList.add('active');
                pipeYOLO.title = `YOLOv8 — ${data.custom_model ? 'Custom ✅' : 'Fallback COCO'}`;
            }

            if (data.fuzzy_engine) {
                pipeFuzzy.classList.add('active');
                pipeFuzzy.title = 'Fuzzy Logic Engine — Ready ✅';
            } else {
                pipeFuzzy.classList.add('unavail');
                pipeFuzzy.title = 'Fuzzy not ready — pip install scikit-fuzzy';
            }

        } catch (e) {
            headerStatus.className = 'header-status';
            headerStatus.querySelector('.status-text').textContent = 'Offline';
        }
    }

    // ─── Upload Handling ───────────────────────────────────────────────────────

    function setupUpload() {
        uploadZone.addEventListener('click', (e) => {
            if (btnNewScan && (e.target === btnNewScan || btnNewScan.contains(e.target))) return;
            fileInput.click();
        });

        uploadZone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) processFile(file);
        });

        uploadZone.addEventListener('dragenter', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
        uploadZone.addEventListener('dragover',  (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            if (!uploadZone.contains(e.relatedTarget)) uploadZone.classList.remove('drag-over');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) processFile(file);
        });

        btnNewScan.addEventListener('click', (e) => { e.stopPropagation(); resetView(); });
    }

    function processFile(file) {
        if (!file.type.startsWith('image/')) { showError('Please upload an image file (JPEG, PNG, WebP).'); return; }
        if (file.size > MAX_FILE_SIZE)        { showError('File too large. Maximum size is 10MB.');          return; }
        uploadImage(file);
    }

    // ─── API Call ──────────────────────────────────────────────────────────────

    async function uploadImage(file) {
        showLoading(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_BASE}/api/detect`, { method: 'POST', body: formData });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${res.status}`);
            }

            const data = await res.json();
            if (data.success) {
                renderResults(data, file.name);
            } else {
                throw new Error('Detection failed');
            }
        } catch (err) {
            showError(err.message || 'Failed to connect to the server.');
            showLoading(false);
        }
    }

    // ─── Render Results ────────────────────────────────────────────────────────

    function renderResults(data, filename) {
        showLoading(false);

        heroSection.style.display      = 'none';
        uploadSection.style.display    = 'none';
        categoriesSection.style.display = 'none';
        resultsSection.classList.remove('hidden');

        // Summary bar
        const [w, h] = data.summary.image_size;
        totalDetections.textContent = data.summary.total_detections;
        inferenceTime.textContent   = `${data.summary.inference_time_ms}ms`;
        imageSize.textContent       = `${w}×${h}`;
        panelFilename.textContent   = truncateFilename(filename, 30);
        panelCount.textContent      = `${data.summary.total_detections} pest${data.summary.total_detections !== 1 ? 's' : ''}`;

        // CNN Global Banner
        if (data.summary.cnn_loaded && data.summary.cnn_top_class) {
            cnnGlobalBanner.classList.remove('hidden');
            cnnGlobalResult.textContent = data.summary.cnn_top_class;
            cnnGlobalConf.textContent   = `${(data.summary.cnn_top_confidence * 100).toFixed(1)}%`;
        } else {
            cnnGlobalBanner.classList.add('hidden');
        }

        // Images
        originalImage.src  = data.original_image;
        annotatedImage.src = data.annotated_image;

        // Detection cards
        detectionsGrid.innerHTML = '';

        if (data.detections.length === 0) {
            noDetections.classList.remove('hidden');
            detectionsGrid.style.display = 'none';
        } else {
            noDetections.classList.add('hidden');
            detectionsGrid.style.display = '';
            data.detections.forEach((det, i) => {
                detectionsGrid.appendChild(createDetectionCard(det, i));
            });

            // Animate bars after render
            requestAnimationFrame(() => {
                setTimeout(() => {
                    document.querySelectorAll('.det-bar-fill').forEach((bar) => {
                        bar.style.width = bar.dataset.width;
                    });
                }, 100);
            });
        }

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    /**
     * Creates a detection card showing all 3 model scores independently.
     */
    function createDetectionCard(det, index) {
        const card = document.createElement('div');
        card.className = 'detection-card hybrid-card';
        card.style.setProperty('--det-color', det.color);
        card.style.animationDelay = `${index * 0.08}s`;

        const yoloPct      = ((det.yolo_conf  || 0) * 100).toFixed(1);
        const cnnPct       = ((det.cnn_prob   || 0) * 100).toFixed(1);
        const combinedPct  = ((det.combined_conf || det.confidence || 0) * 100).toFixed(1);
        const fuzzyLabel   = det.fuzzy_severity || det.severity || 'N/A';
        const fuzzyScore   = det.fuzzy_score != null ? det.fuzzy_score.toFixed(1) : '—';
        const bboxStr      = det.bbox.map((v) => Math.round(v)).join(', ');
        const severityClass = getSeverityClass(fuzzyLabel);

        card.innerHTML = `
            <!-- Card Header -->
            <div class="det-header">
                <div class="det-name">
                    <span class="det-icon">${det.icon || '🔍'}</span>
                    <span class="det-label">${escapeHtml(det.class_name)}</span>
                </div>
                <span class="det-combined-badge" title="60% YOLO + 40% CNN combined score">${combinedPct}%</span>
            </div>

            <!-- 3 Model Score Rows -->
            <div class="model-scores">

                <!-- YOLO Score -->
                <div class="ms-row" title="YOLOv8 detection confidence">
                    <div class="ms-label">
                        <span class="ms-dot yolo"></span>
                        <span class="ms-name">YOLOv8</span>
                    </div>
                    <div class="ms-bar-track">
                        <div class="det-bar-fill ms-bar yolo-bar" data-width="${yoloPct}%" style="width: 0%"></div>
                    </div>
                    <span class="ms-value">${yoloPct}%</span>
                </div>

                <!-- CNN Score -->
                <div class="ms-row" title="MobileNetV2 classification probability">
                    <div class="ms-label">
                        <span class="ms-dot cnn"></span>
                        <span class="ms-name">CNN</span>
                    </div>
                    <div class="ms-bar-track">
                        <div class="det-bar-fill ms-bar cnn-bar" data-width="${cnnPct}%" style="width: 0%"></div>
                    </div>
                    <span class="ms-value">${cnnPct}%</span>
                </div>

                <!-- Fuzzy Severity -->
                <div class="ms-row fuzzy-row" title="Fuzzy Logic severity score (0–100)">
                    <div class="ms-label">
                        <span class="ms-dot fuzzy"></span>
                        <span class="ms-name">Fuzzy</span>
                    </div>
                    <div class="ms-bar-track">
                        <div class="det-bar-fill ms-bar fuzzy-bar" data-width="${fuzzyScore}%" style="width: 0%"></div>
                    </div>
                    <span class="ms-value fuzzy-score-val">${fuzzyScore}/100</span>
                </div>
            </div>

            <!-- Footer meta -->
            <div class="det-footer">
                <span class="det-tag" title="Bounding box coordinates">📍 [${bboxStr}]</span>
                <span class="det-severity-badge ${severityClass}">${fuzzyLabel}</span>
            </div>
        `;

        return card;
    }

    function getSeverityClass(label) {
        if (!label) return 'medium';
        const l = label.toLowerCase();
        if (l.includes('high') || l.includes('critical')) return 'high';
        if (l.includes('low'))                             return 'low';
        return 'medium';
    }

    // ─── Loading Stages Animation ──────────────────────────────────────────────

    let loadingTimer = null;

    function showLoading(show) {
        if (show) {
            uploadZone.classList.add('loading');
            uploadLoading.classList.remove('hidden');
            // Animate through stages
            [lsCNN, lsYOLO, lsFuzzy].forEach(el => el && el.classList.remove('ls-active', 'ls-done'));
            let stage = 0;
            const stages = [lsCNN, lsYOLO, lsFuzzy];
            loadingTimer = setInterval(() => {
                if (stage > 0 && stages[stage - 1]) stages[stage - 1].classList.replace('ls-active', 'ls-done');
                if (stage < stages.length && stages[stage]) stages[stage].classList.add('ls-active');
                stage++;
                if (stage >= stages.length) clearInterval(loadingTimer);
            }, 600);
        } else {
            uploadZone.classList.remove('loading');
            uploadLoading.classList.add('hidden');
            if (loadingTimer) clearInterval(loadingTimer);
        }
    }

    function resetView() {
        resultsSection.classList.add('hidden');
        heroSection.style.display       = '';
        uploadSection.style.display     = '';
        categoriesSection.style.display = '';
        fileInput.value = '';
        uploadSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function showError(message) {
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
            background: #1e293b; color: #f1f5f9; padding: 14px 24px;
            border-radius: 12px; font-size: 0.9rem; font-weight: 500;
            border: 1px solid rgba(239,68,68,0.3); box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            z-index: 1000; display: flex; align-items: center; gap: 10px;
            animation: fadeUp 0.3s ease; font-family: 'Inter', sans-serif;
        `;
        toast.innerHTML = `<span style="color:#ef4444;font-size:1.2rem;">⚠</span> ${escapeHtml(message)}`;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.transition  = 'opacity 0.3s, transform 0.3s';
            toast.style.opacity     = '0';
            toast.style.transform   = 'translateX(-50%) translateY(10px)';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    function truncateFilename(name, max) {
        if (name.length <= max) return name;
        const ext = name.substring(name.lastIndexOf('.'));
        return name.substring(0, max - ext.length - 3) + '...' + ext;
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ─── Initialize ────────────────────────────────────────────────────────────

    function init() {
        setupUpload();
        checkHealth();
        setInterval(checkHealth, 30000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();

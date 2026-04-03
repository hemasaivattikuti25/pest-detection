/**
 * F-YOLO PestVision — Frontend Application
 * Handles image upload, API communication, and results rendering.
 * RAW images only — no client-side preprocessing.
 */

(function () {
    'use strict';

    // ─── DOM Elements ─────────────────────────────────────────────────────

    const headerStatus = document.getElementById('headerStatus');
    const uploadZone = document.getElementById('uploadZone');
    const uploadIcon = document.getElementById('uploadIcon');
    const uploadLoading = document.getElementById('uploadLoading');
    const fileInput = document.getElementById('fileInput');
    const heroSection = document.getElementById('heroSection');
    const uploadSection = document.getElementById('uploadSection');
    const resultsSection = document.getElementById('resultsSection');
    const categoriesSection = document.getElementById('categoriesSection');

    // Results elements
    const totalDetections = document.getElementById('totalDetections');
    const inferenceTime = document.getElementById('inferenceTime');
    const imageSize = document.getElementById('imageSize');
    const btnNewScan = document.getElementById('btnNewScan');
    const originalImage = document.getElementById('originalImage');
    const annotatedImage = document.getElementById('annotatedImage');
    const panelFilename = document.getElementById('panelFilename');
    const panelCount = document.getElementById('panelCount');
    const detectionsGrid = document.getElementById('detectionsGrid');
    const noDetections = document.getElementById('noDetections');

    const API_BASE = '';
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

    // ─── Health Check ─────────────────────────────────────────────────────

    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            const data = await res.json();

            if (data.status === 'healthy' && data.model_loaded) {
                if (data.custom_model) {
                    headerStatus.className = 'header-status online';
                    headerStatus.querySelector('.status-text').textContent = 'Model Ready';
                } else {
                    headerStatus.className = 'header-status fallback';
                    headerStatus.querySelector('.status-text').textContent = 'Fallback Model';
                }
            } else {
                headerStatus.className = 'header-status';
                headerStatus.querySelector('.status-text').textContent = 'Model Error';
            }
        } catch (e) {
            headerStatus.className = 'header-status';
            headerStatus.querySelector('.status-text').textContent = 'Offline';
        }
    }

    // ─── Upload Handling ──────────────────────────────────────────────────

    function setupUpload() {
        // Click to browse
        uploadZone.addEventListener('click', (e) => {
            if (e.target === btnNewScan || btnNewScan.contains(e.target)) return;
            fileInput.click();
        });

        // Keyboard accessibility
        uploadZone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });

        // File selected
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) processFile(file);
        });

        // Drag and drop
        uploadZone.addEventListener('dragenter', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });

        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            if (!uploadZone.contains(e.relatedTarget)) {
                uploadZone.classList.remove('drag-over');
            }
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) processFile(file);
        });

        // New scan button
        btnNewScan.addEventListener('click', (e) => {
            e.stopPropagation();
            resetView();
        });
    }

    function processFile(file) {
        // Validate type
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file (JPEG, PNG, WebP).');
            return;
        }

        // Validate size
        if (file.size > MAX_FILE_SIZE) {
            showError('File too large. Maximum size is 10MB.');
            return;
        }

        uploadImage(file);
    }

    // ─── API Call ─────────────────────────────────────────────────────────

    async function uploadImage(file) {
        showLoading(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_BASE}/api/detect`, {
                method: 'POST',
                body: formData,
            });

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

    // ─── Render Results ───────────────────────────────────────────────────

    function renderResults(data, filename) {
        showLoading(false);

        // Hide hero, show results
        heroSection.style.display = 'none';
        uploadSection.style.display = 'none';
        categoriesSection.style.display = 'none';
        resultsSection.classList.remove('hidden');

        // Summary
        totalDetections.textContent = data.summary.total_detections;
        inferenceTime.textContent = `${data.summary.inference_time_ms}ms`;
        const [w, h] = data.summary.image_size;
        imageSize.textContent = `${w}×${h}`;
        panelFilename.textContent = truncateFilename(filename, 30);
        panelCount.textContent = `${data.summary.total_detections} pest${data.summary.total_detections !== 1 ? 's' : ''}`;

        // Images
        originalImage.src = data.original_image;
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
                const card = createDetectionCard(det, i);
                detectionsGrid.appendChild(card);
            });

            // Animate confidence bars after render
            requestAnimationFrame(() => {
                setTimeout(() => {
                    document.querySelectorAll('.det-bar-fill').forEach((bar) => {
                        bar.style.width = bar.dataset.width;
                    });
                }, 100);
            });
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function createDetectionCard(det, index) {
        const card = document.createElement('div');
        card.className = 'detection-card';
        card.style.setProperty('--det-color', det.color);
        card.style.animationDelay = `${index * 0.08}s`;

        const confidencePct = (det.confidence * 100).toFixed(1);
        const bboxStr = det.bbox.map((v) => Math.round(v)).join(', ');
        const severityClass = (det.severity || 'medium').toLowerCase();

        card.innerHTML = `
            <div class="det-header">
                <div class="det-name">
                    <span class="det-icon">${det.icon || '🔍'}</span>
                    <span class="det-label">${det.class_name}</span>
                </div>
                <span class="det-confidence-value">${confidencePct}%</span>
            </div>
            <div class="det-bar-track">
                <div class="det-bar-fill" data-width="${confidencePct}%" style="width: 0%"></div>
            </div>
            <div class="det-meta">
                <span class="det-tag">bbox: [${bboxStr}]</span>
                ${det.severity && det.severity !== 'N/A' ? `<span class="det-severity ${severityClass}">${det.severity}</span>` : ''}
            </div>
        `;

        return card;
    }

    // ─── UI State Helpers ─────────────────────────────────────────────────

    function showLoading(show) {
        if (show) {
            uploadZone.classList.add('loading');
            uploadLoading.classList.remove('hidden');
        } else {
            uploadZone.classList.remove('loading');
            uploadLoading.classList.add('hidden');
        }
    }

    function resetView() {
        resultsSection.classList.add('hidden');
        heroSection.style.display = '';
        uploadSection.style.display = '';
        categoriesSection.style.display = '';

        // Reset file input
        fileInput.value = '';

        // Scroll to upload
        uploadSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function showError(message) {
        // Simple error — create a toast notification
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
            background: #1e293b; color: #f1f5f9; padding: 14px 24px;
            border-radius: 12px; font-size: 0.9rem; font-weight: 500;
            border: 1px solid rgba(239,68,68,0.3); box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            z-index: 1000; display: flex; align-items: center; gap: 10px;
            animation: fadeUp 0.3s ease;
            font-family: 'Inter', sans-serif;
        `;
        toast.innerHTML = `<span style="color:#ef4444;font-size:1.2rem;">⚠</span> ${escapeHtml(message)}`;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.transition = 'opacity 0.3s, transform 0.3s';
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(10px)';
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

    // ─── Initialize ───────────────────────────────────────────────────────

    function init() {
        setupUpload();
        checkHealth();
        // Re-check health every 30s
        setInterval(checkHealth, 30000);
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();

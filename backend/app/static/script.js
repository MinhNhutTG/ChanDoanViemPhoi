document.addEventListener('DOMContentLoaded', () => {
    const dropzone       = document.getElementById('dropzone');
    const fileInput      = document.getElementById('fileInput');
    const imagePreview   = document.getElementById('imagePreview');
    const uploadPrompt   = document.getElementById('upload-prompt');
    const analyzeBtn     = document.getElementById('analyzeBtn');
    const resultContent  = document.getElementById('result-content');
    const noResult       = document.getElementById('no-result');
    const gradcamImage   = document.getElementById('gradcam-image');
    const predictionText = document.getElementById('prediction-text');
    const resultBadge    = document.getElementById('result-badge');
    const probBars       = document.getElementById('prob-bars');
    const reportContent  = document.getElementById('report-content');

    // ── 1. MỞ CHỌN FILE ──────────────────────────────────────────────
    dropzone.addEventListener('click', () => fileInput.click());

    // ── 2. PREVIEW ẢNH ───────────────────────────────────────────────
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/preview', { method: 'POST', body: formData });
            const data = await response.json();

            imagePreview.src = 'data:image/jpeg;base64,' + data.image;
            imagePreview.classList.remove('hidden');
            uploadPrompt.classList.add('hidden');

            // Reset về trạng thái chờ
            resultContent.classList.add('hidden');
            noResult.classList.remove('hidden');
        } catch (err) {
            console.error('Preview error:', err);
        }
    });

    // ── 3. DỰ ĐOÁN ───────────────────────────────────────────────────
    window.predict = async function () {
        if (!fileInput.files[0]) {
            alert('Vui lòng chọn một file ảnh X-ray trước!');
            return;
        }

        // Trạng thái loading
        const originalBtnHTML = analyzeBtn.innerHTML;
        analyzeBtn.innerHTML  = '<i class="fas fa-spinner fa-spin mr-2"></i>Đang phân tích...';
        analyzeBtn.disabled   = true;

        noResult.classList.add('hidden');
        resultContent.classList.remove('hidden');
        predictionText.innerText = 'Đang xử lý...';
        _resetBadgeStyle();

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        try {
            const response = await fetch('/api/predict', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Lỗi Server ' + response.status);

            const data        = await response.json();
            const label       = data.prediction.prediction;
            const confidence  = data.prediction.confidence;
            const probs       = data.prediction.probabilities || {};
            const report      = data.prediction.report || '';

            // ── Tiêu đề kết quả ──
            predictionText.innerText = _formatLabel(label);

            // ── Style badge theo loại kết quả ──
            _applyBadgeStyle(label);

            // ── Probability bars ──
            _renderProbBars(probs, confidence, label);

            // ── Báo cáo chi tiết ──
            _renderReport(report);

            // ── GradCAM ──
            if (data.prediction.heatmap_url) {
                gradcamImage.src = data.prediction.heatmap_url;
                gradcamImage.style.animation = 'none';
                requestAnimationFrame(() => {
                    gradcamImage.style.animation = 'fadeSlideUp 0.5s cubic-bezier(.22,1,.36,1) both';
                });
            }

        } catch (err) {
            console.error(err);
            predictionText.innerText = 'Lỗi kết nối';
            _applyErrorStyle();
        } finally {
            analyzeBtn.innerHTML = originalBtnHTML;
            analyzeBtn.disabled  = false;
        }
    };

    // ════════════════════════════════════════════════════
    //  HELPER FUNCTIONS
    // ════════════════════════════════════════════════════

    /** Chuyển nhãn model → tên hiển thị tiếng Việt */
    function _formatLabel(label) {
        const lbl = (label || '').toUpperCase();
        if (lbl.includes('MỜ PHỔI'))  return 'Mờ phổi (Lung Opacity)';
        if (lbl.includes('BÌNH THƯỜNG'))        return 'Bình thường (Normal)';
        return label;
    }

    /** Reset badge về trạng thái trung tính khi đang chờ */
    function _resetBadgeStyle() {
        resultBadge.className = 'p-4 rounded-2xl';
        resultBadge.style.animation = '';
        predictionText.className    = 'display font-normal uppercase';
        predictionText.style.cssText = 'font-size:2.2rem; color:var(--text-dark); letter-spacing:-0.01em;';
    }

    /** Áp dụng style badge + màu chữ theo kết quả */
    function _applyBadgeStyle(label) {
        const lbl = (label || '').toUpperCase();

        // Animation mỗi lần đổi kết quả
        resultBadge.style.animation = 'none';
        requestAnimationFrame(() => {
            resultBadge.style.animation = 'fadeSlideUp 0.45s cubic-bezier(.22,1,.36,1) both';
        });

        predictionText.className = 'display font-normal uppercase';

        if (lbl.includes('MỜ PHỔI') || lbl.includes('VIÊM PHỔI')) {
            // 🔴 Mờ phổi
            resultBadge.className        = 'badge-opacity p-4';
            predictionText.style.cssText = 'font-size:2.2rem; color:#e11d48; letter-spacing:-0.01em;';

        } else {
            // 🟢 Bình thường
            resultBadge.className        = 'badge-normal p-4';
            predictionText.style.cssText = 'font-size:2.2rem; color:#15803d; letter-spacing:-0.01em;';
        }
    }

    /** Style khi xảy ra lỗi kết nối */
    function _applyErrorStyle() {
        resultBadge.className        = 'badge-notnormal p-4';
        predictionText.className     = 'display font-normal uppercase';
        predictionText.style.cssText = 'font-size:2rem; color:#ea580c; letter-spacing:-0.01em;';
    }

    /** Render probability bars cho từng class */
    function _renderProbBars(probs, confidence, predLabel) {
        if (!probs || Object.keys(probs).length === 0) {
            probBars.innerHTML = `
                <div style="color:var(--text-soft); font-size:13px; font-style:italic;">
                    Không có dữ liệu xác suất.
                </div>`;
            return;
        }

        // Màu sắc theo class
        const config = {
            'lung_opacity': { color: '#e11d48', bg: 'rgba(225,29,72,0.08)',  label: 'Mờ phổi'    },
            'normal':       { color: '#15803d', bg: 'rgba(21,128,61,0.08)',  label: 'Bình thường' },
        };

        // Sắp xếp giảm dần theo xác suất
        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

        probBars.innerHTML = sorted.map(([cls, prob]) => {
            const pct  = (prob * 100).toFixed(1);
            const cfg  = config[cls.toLowerCase()] || { color: 'var(--sky-400)', bg: 'rgba(56,189,248,0.08)', label: cls };
            const isTop = cls.toLowerCase() === (predLabel || '').toLowerCase();

            return `
            <div style="margin-bottom: 10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-size:12px; font-weight:${isTop ? '700' : '500'}; color:${isTop ? cfg.color : 'var(--text-mid)'};">
                        ${isTop ? '▶ ' : ''}${cfg.label}
                    </span>
                    <span style="font-family:'Fira Code',monospace; font-size:12px; font-weight:600; color:${cfg.color};">
                        ${pct}%
                    </span>
                </div>
                <div style="width:100%; height:8px; border-radius:999px; background:${cfg.bg}; overflow:hidden;">
                    <div style="
                        height:8px; border-radius:999px;
                        background:${cfg.color};
                        width:0%;
                        transition: width 0.85s cubic-bezier(.22,1,.36,1);
                        opacity:${isTop ? '1' : '0.55'};
                    " data-w="${pct}%"></div>
                </div>
            </div>`;
        }).join('');

        // Animate bars sau khi render
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                probBars.querySelectorAll('[data-w]').forEach(bar => {
                    bar.style.width = bar.dataset.w;
                });
            });
        });
    }

    /** Render báo cáo LLM với hiệu ứng typewriter */
    function _renderReport(report) {
        const badge        = document.getElementById('llm-report-badge');
        const loadingBadge = document.getElementById('llm-loading-badge');

        // Reset trạng thái
        reportContent.classList.remove('done');
        reportContent.classList.add('typing-cursor');
        if (badge)        badge.classList.add('hidden');
        if (loadingBadge) loadingBadge.classList.remove('hidden');

        if (!report || report.trim() === '') {
            reportContent.style.cssText = 'color:var(--text-soft); font-size:13px; font-style:italic;';
            reportContent.innerText = 'Không có báo cáo chi tiết từ hệ thống.';
            reportContent.classList.remove('typing-cursor');
            reportContent.classList.add('done');
            if (badge)        badge.classList.remove('hidden');
            if (loadingBadge) loadingBadge.classList.add('hidden');
            return;
        }

        // Typewriter effect
        reportContent.style.cssText = 'color:var(--text-mid); font-size:13px; line-height:1.8;';
        reportContent.innerText = '';

        const lines   = report.split('\n');
        let lineIdx   = 0;
        let charIdx   = 0;
        const speed   = 14; // ms mỗi ký tự

        function typeNext() {
            if (lineIdx >= lines.length) {
                // Hoàn tất
                reportContent.classList.remove('typing-cursor');
                reportContent.classList.add('done');
                if (badge)        badge.classList.remove('hidden');
                if (loadingBadge) loadingBadge.classList.add('hidden');
                return;
            }
            const line = lines[lineIdx];
            if (charIdx < line.length) {
                reportContent.innerText += line[charIdx];
                charIdx++;
                setTimeout(typeNext, speed);
            } else {
                // Xuống dòng
                if (lineIdx < lines.length - 1) reportContent.innerText += '\n';
                lineIdx++;
                charIdx = 0;
                setTimeout(typeNext, speed * 3); // nhịp nghỉ giữa dòng
            }
            // Auto-scroll xuống
            const body = reportContent.closest('.llm-report-body');
            if (body) body.scrollTop = body.scrollHeight;
        }

        // Delay nhỏ trước khi bắt đầu gõ
        setTimeout(typeNext, 300);
    }
});
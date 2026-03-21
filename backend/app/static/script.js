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

    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        try {
            const res  = await fetch('/api/preview', { method: 'POST', body: formData });
            const data = await res.json();
            imagePreview.src = 'data:image/jpeg;base64,' + data.image;
            imagePreview.classList.remove('hidden');
            uploadPrompt.classList.add('hidden');
            resultContent.classList.add('hidden');
            noResult.classList.remove('hidden');
        } catch (err) { console.error(err); }
    });

    window.predict = async function () {
        if (!fileInput.files[0]) { alert('Vui lòng chọn file ảnh X-ray!'); return; }

        const orig = analyzeBtn.innerHTML;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Đang phân tích...';
        analyzeBtn.disabled  = true;

        noResult.classList.add('hidden');
        resultContent.classList.remove('hidden');
        predictionText.innerText = 'Đang xử lý...';
        _resetBadge();

        const fd = new FormData();
        fd.append('image', fileInput.files[0]);

        try {
            const res = await fetch('/api/predict', { method: 'POST', body: fd });
            if (!res.ok) throw new Error('Server error ' + res.status);
            const data       = await res.json();
            const label      = data.prediction.prediction;
            const confidence = data.prediction.confidence;
            const probs      = data.prediction.probabilities || {};
            const report     = data.prediction.report || '';

            predictionText.innerText = _fmt(label);
            _badge(label);
            _bars(probs, confidence, label);
            _report(report);

            if (data.prediction.heatmap_url) {
                gradcamImage.src = data.prediction.heatmap_url;
                gradcamImage.style.animation = 'none';
                requestAnimationFrame(() => {
                    gradcamImage.style.animation = 'fadeUp .5s ease both';
                });
            }
        } catch (err) {
            console.error(err);
            predictionText.innerText = 'Lỗi kết nối';
            _badgeError();
        } finally {
            analyzeBtn.innerHTML = orig;
            analyzeBtn.disabled  = false;
        }
    };

    function _fmt(label) {
        const l = (label || '').toUpperCase();
        if (l.includes('MỜ PHỔI'))    return 'Mờ phổi (Lung Opacity)';
        if (l.includes('BÌNH THƯỜNG')) return 'Bình thường (Normal)';
        return label;
    }

    function _resetBadge() {
        resultBadge.className = 'p-4';
        resultBadge.style.cssText = '';
        predictionText.style.cssText = 'font-size:2rem; color:var(--text-1);';
    }

    function _badge(label) {
        const l = (label || '').toUpperCase();
        resultBadge.style.animation = 'none';
        requestAnimationFrame(() => {
            resultBadge.style.animation = 'fadeUp .4s ease both';
        });
        if (l.includes('MỜ PHỔI') || l.includes('VIÊM PHỔI')) {
            resultBadge.className        = 'badge-opacity p-4';
            predictionText.style.cssText = 'font-size:2rem; color:var(--red-600);';
        } else {
            resultBadge.className        = 'badge-normal p-4';
            predictionText.style.cssText = 'font-size:2rem; color:var(--green-600);';
        }
        predictionText.classList.add('serif');
    }

    function _badgeError() {
        resultBadge.className        = 'badge-notnormal p-4';
        predictionText.style.cssText = 'font-size:1.8rem; color:var(--amber-500);';
        predictionText.classList.add('serif');
    }

    function _bars(probs, confidence, predLabel) {
        if (!probs || Object.keys(probs).length === 0) {
            probBars.innerHTML = '<div style="color:var(--text-4);font-size:12px;font-style:italic;">Không có dữ liệu xác suất.</div>';
            return;
        }
        const cfg = {
            'lung_opacity': { color: '#ef4444', track: '#fee2e2', label: 'Mờ phổi' },
            'normal':       { color: '#22c55e', track: '#dcfce7', label: 'Bình thường' },
        };
        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        probBars.innerHTML = sorted.map(([cls, prob]) => {
            const pct  = (prob * 100).toFixed(1);
            const c    = cfg[cls.toLowerCase()] || { color: '#3b82f6', track: '#dbeafe', label: cls };
            const top  = cls.toLowerCase() === (predLabel || '').toLowerCase();
            return `
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <span style="font-size:12px;font-weight:${top?'700':'500'};color:${top?c.color:'var(--text-3)'};">
                        ${top ? '▶ ' : ''}${c.label}
                    </span>
                    <span style="font-family:'Fira Code',monospace;font-size:11px;font-weight:600;color:${c.color};">${pct}%</span>
                </div>
                <div style="width:100%;height:6px;border-radius:999px;background:${c.track};overflow:hidden;">
                    <div style="height:6px;border-radius:999px;background:${c.color};width:0%;transition:width .9s cubic-bezier(.22,1,.36,1);opacity:${top?'1':'0.5'};" data-w="${pct}%"></div>
                </div>
            </div>`;
        }).join('');
        requestAnimationFrame(() => requestAnimationFrame(() => {
            probBars.querySelectorAll('[data-w]').forEach(b => { b.style.width = b.dataset.w; });
        }));
    }

    function _report(report) {
        const badge   = document.getElementById('llm-report-badge');
        const loading = document.getElementById('llm-loading-badge');
        reportContent.classList.remove('done');
        reportContent.classList.add('typing-cursor');
        badge?.classList.add('hidden');
        loading?.classList.remove('hidden');

        if (!report || !report.trim()) {
            reportContent.style.cssText = 'color:var(--text-4);font-size:13px;font-style:italic;';
            reportContent.innerText = 'Không có báo cáo chi tiết.';
            reportContent.classList.remove('typing-cursor');
            reportContent.classList.add('done');
            badge?.classList.remove('hidden');
            loading?.classList.add('hidden');
            return;
        }

        reportContent.style.cssText = 'color:var(--text-2);font-size:13px;line-height:1.75;';
        reportContent.innerText = '';
        const lines = report.split('\n');
        let li = 0, ci = 0;

        function type() {
            if (li >= lines.length) {
                reportContent.classList.remove('typing-cursor');
                reportContent.classList.add('done');
                badge?.classList.remove('hidden');
                loading?.classList.add('hidden');
                return;
            }
            const line = lines[li];
            if (ci < line.length) {
                reportContent.innerText += line[ci++];
                setTimeout(type, 13);
            } else {
                if (li < lines.length - 1) reportContent.innerText += '\n';
                li++; ci = 0;
                setTimeout(type, 40);
            }
            const body = reportContent.closest('.report-body');
            if (body) body.scrollTop = body.scrollHeight;
        }
        setTimeout(type, 300);
    }
});
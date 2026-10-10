/**
 * SignVerse AI - Real-Time Sign Language Translator Frontend Client
 * Synchronizes with Flask Backend Stream & Real-Time Machine Learning Pipeline
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const streamVideo = document.getElementById('streamVideo');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const startCamBtn = document.getElementById('startCamBtn');
    const stopCamBtn = document.getElementById('stopCamBtn');

    const confSlider = document.getElementById('confSlider');
    const confValText = document.getElementById('confVal');
    const holdSlider = document.getElementById('holdSlider');
    const holdValText = document.getElementById('holdVal');

    const signDisplay = document.getElementById('signDisplay');
    const confBadge = document.getElementById('confBadge');
    const holdCountText = document.getElementById('holdCountText');
    const holdProgressBar = document.getElementById('holdProgressBar');

    const sentenceBox = document.getElementById('sentenceBox');
    const speakBtn = document.getElementById('speakBtn');
    const copyBtn = document.getElementById('copyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const spaceBtn = document.getElementById('spaceBtn');
    const backspaceBtn = document.getElementById('backspaceBtn');

    const probsList = document.getElementById('probsList');
    const ttsToggleBtn = document.getElementById('ttsToggleBtn');
    const ttsStatus = document.getElementById('ttsStatus');
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const refModalBtn = document.getElementById('refModalBtn');
    const guideModal = document.getElementById('guideModal');
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const aslGrid = document.getElementById('aslGrid');
    const toast = document.getElementById('toast');

    // App State Variables
    let isTTSEnabled = true;
    let previousSentence = "";

    // Sliders
    confSlider.addEventListener('input', (e) => {
        confValText.textContent = `${e.target.value}%`;
    });

    holdSlider.addEventListener('input', (e) => {
        holdValText.textContent = `${e.target.value} frames`;
    });

    // Theme Toggle
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        themeToggleBtn.innerHTML = newTheme === 'dark' ? '<i class="fa-solid fa-moon"></i>' : '<i class="fa-solid fa-sun"></i>';
    });

    // Audio Speech Toggle
    ttsToggleBtn.addEventListener('click', () => {
        isTTSEnabled = !isTTSEnabled;
        ttsStatus.textContent = isTTSEnabled ? 'Audio ON' : 'Audio OFF';
        ttsToggleBtn.style.opacity = isTTSEnabled ? '1' : '0.6';
        showToast(isTTSEnabled ? 'Auto Speech Enabled' : 'Auto Speech Muted');
    });

    // Stream Controls
    startCamBtn.addEventListener('click', () => {
        streamVideo.src = "/video_feed?" + new Date().getTime();
        streamVideo.style.display = 'block';
        videoPlaceholder.style.display = 'none';
        startCamBtn.disabled = true;
        stopCamBtn.disabled = false;
        showToast("Live Stream Started");
    });

    stopCamBtn.addEventListener('click', () => {
        streamVideo.src = "";
        streamVideo.style.display = 'none';
        videoPlaceholder.style.display = 'flex';
        startCamBtn.disabled = false;
        stopCamBtn.disabled = true;
        showToast("Live Stream Paused");
    });

    // Poll Backend State every 100ms
    setInterval(async () => {
        try {
            const res = await fetch('/api/state');
            if (res.ok) {
                const state = await res.json();
                updateUIState(state);
            }
        } catch (err) {
            // Ignore temporary polling errors
        }
    }, 100);

    function updateUIState(state) {
        // 1. Update Sentence
        const currentSentence = state.sentence || "";
        if (currentSentence !== previousSentence) {
            // New character added
            if (isTTSEnabled && currentSentence.length > previousSentence.length) {
                const addedChar = currentSentence.slice(-1);
                if (addedChar.trim()) speakText(addedChar);
            }
            previousSentence = currentSentence;
            updateSentenceDisplay(currentSentence);
        }

        // 2. Update Spotlight Sign & Confidence
        const sign = state.current_sign || "None";
        const conf = state.confidence || 0.0;
        signDisplay.textContent = sign !== "None" ? sign : "?";
        confBadge.textContent = `Confidence: ${(conf * 100).toFixed(0)}%`;

        if (conf > 0.8) {
            signDisplay.style.color = 'var(--primary)';
        } else if (conf > 0.5) {
            signDisplay.style.color = 'var(--warning)';
        } else {
            signDisplay.style.color = 'var(--text-muted)';
        }

        // 3. Update Lock-in Hold Progress Bar
        const framesHeld = state.frames_held || 0;
        const requiredFrames = state.required_frames || 8;
        const ratio = Math.min(Math.max(framesHeld / requiredFrames, 0), 1);
        
        holdProgressBar.style.width = `${ratio * 100}%`;
        holdCountText.textContent = `${framesHeld} / ${requiredFrames} frames`;

        // 4. Update Candidate Probabilities List
        if (state.probs && Object.keys(state.probs).length > 0) {
            updateProbabilitiesList(state.probs);
        }
    }

    function updateSentenceDisplay(text) {
        if (!text || text.trim() === "") {
            sentenceBox.innerHTML = `<span class="placeholder-text">Translated text will appear here as you sign...</span>`;
        } else {
            sentenceBox.textContent = text;
        }
    }

    function updateProbabilitiesList(probs) {
        const entries = Object.entries(probs);
        if (entries.length === 0) return;

        probsList.innerHTML = entries.map(([s, p]) => `
            <div class="prob-row">
                <span class="label">${s}</span>
                <div class="bar-bg"><div class="bar-fill" style="width: ${(p * 100).toFixed(0)}%"></div></div>
                <span class="val">${(p * 100).toFixed(0)}%</span>
            </div>
        `).join('');
    }

    // Action API Controls
    clearBtn.addEventListener('click', async () => {
        await fetch('/api/clear', { method: 'POST' });
        previousSentence = "";
        updateSentenceDisplay("");
        showToast("Sentence Cleared");
    });

    backspaceBtn.addEventListener('click', async () => {
        const res = await fetch('/api/backspace', { method: 'POST' });
        if (res.ok) {
            const data = await res.json();
            previousSentence = data.sentence;
            updateSentenceDisplay(data.sentence);
        }
    });

    spaceBtn.addEventListener('click', async () => {
        const res = await fetch('/api/space', { method: 'POST' });
        if (res.ok) {
            const data = await res.json();
            previousSentence = data.sentence;
            updateSentenceDisplay(data.sentence);
        }
    });

    speakBtn.addEventListener('click', () => {
        if (previousSentence.trim()) speakText(previousSentence);
        else showToast("Sentence is empty");
    });

    copyBtn.addEventListener('click', () => {
        if (previousSentence.trim()) {
            navigator.clipboard.writeText(previousSentence);
            showToast("Copied to clipboard!");
        } else {
            showToast("Sentence is empty");
        }
    });

    // Modal Controls
    refModalBtn.addEventListener('click', () => {
        populateASLGrid();
        guideModal.classList.add('active');
    });

    modalCloseBtn.addEventListener('click', () => {
        guideModal.classList.remove('active');
    });

    function populateASLGrid() {
        const letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','del'];
        aslGrid.innerHTML = letters.map(l => `<div class="asl-card">${l}</div>`).join('');
    }

    // Speech Synthesis
    function speakText(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    }

    function showToast(msg) {
        toast.textContent = msg;
        toast.classList.add('active');
        setTimeout(() => toast.classList.remove('active'), 2500);
    }
});

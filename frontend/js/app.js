/**
 * GANPAN - AI 쇼츠 영상 제작기
 * Frontend Application
 */

// API URL 설정 (같은 서버에서 서빙)
const API_BASE_URL = '/api';

// State
const state = {
    currentStep: 1,
    images: [],
    story: '',
    style: 'cinematic',
    music: 'calm',
    generatedVideo: null,
    generatedScript: null
};

// DOM Elements
const elements = {
    // Steps
    stepIndicators: document.querySelectorAll('.step'),
    step1: document.getElementById('step1'),
    step2: document.getElementById('step2'),
    step3: document.getElementById('step3'),

    // Upload
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    previewContainer: document.getElementById('previewContainer'),

    // Navigation
    nextToStep2: document.getElementById('nextToStep2'),
    backToStep1: document.getElementById('backToStep1'),
    backToStep2: document.getElementById('backToStep2'),
    generateBtn: document.getElementById('generateBtn'),
    downloadBtn: document.getElementById('downloadBtn'),

    // Story Input
    storyInput: document.getElementById('storyInput'),
    charCount: document.getElementById('charCount'),
    styleChips: document.getElementById('styleChips'),
    musicChips: document.getElementById('musicChips'),

    // Progress
    progressView: document.getElementById('progressView'),
    resultView: document.getElementById('resultView'),
    progressCircle: document.getElementById('progressCircle'),
    progressPercent: document.getElementById('progressPercent'),
    stepAnalyze: document.getElementById('stepAnalyze'),
    stepScript: document.getElementById('stepScript'),
    stepVideo: document.getElementById('stepVideo'),

    // Result
    resultVideo: document.getElementById('resultVideo'),
    generatedScript: document.getElementById('generatedScript'),

    // Snackbar
    snackbar: document.getElementById('snackbar')
};

// Initialize
function init() {
    setupEventListeners();
}

// Event Listeners
function setupEventListeners() {
    // Upload Area
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Navigation
    elements.nextToStep2.addEventListener('click', () => goToStep(2));
    elements.backToStep1.addEventListener('click', () => goToStep(1));
    elements.backToStep2.addEventListener('click', resetAndGoToStep2);
    elements.generateBtn.addEventListener('click', generateVideo);
    elements.downloadBtn.addEventListener('click', downloadVideo);

    // Story Input
    elements.storyInput.addEventListener('input', handleStoryInput);

    // Chips
    elements.styleChips.addEventListener('click', (e) => handleChipSelect(e, 'style'));
    elements.musicChips.addEventListener('click', (e) => handleChipSelect(e, 'music'));

    // Snackbar
    elements.snackbar.querySelector('.snackbar-action').addEventListener('click', hideSnackbar);
}

// File Handling
function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    addImages(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addImages(files);
    e.target.value = ''; // Reset input
}

function addImages(files) {
    const maxImages = 10;
    const maxSize = 10 * 1024 * 1024; // 10MB

    for (const file of files) {
        if (state.images.length >= maxImages) {
            showSnackbar(`최대 ${maxImages}장까지 업로드할 수 있습니다`);
            break;
        }

        if (file.size > maxSize) {
            showSnackbar(`${file.name}: 파일 크기가 10MB를 초과합니다`);
            continue;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            state.images.push({
                file: file,
                dataUrl: e.target.result
            });
            renderPreviews();
            updateNextButton();
        };
        reader.readAsDataURL(file);
    }
}

function removeImage(index) {
    state.images.splice(index, 1);
    renderPreviews();
    updateNextButton();
}

function renderPreviews() {
    elements.previewContainer.innerHTML = state.images.map((img, index) => `
        <div class="preview-item">
            <img src="${img.dataUrl}" alt="Preview ${index + 1}">
            <button class="remove-btn" onclick="removeImage(${index})">
                <span class="material-icons">close</span>
            </button>
            <span class="image-number">${index + 1}</span>
        </div>
    `).join('');
}

function updateNextButton() {
    elements.nextToStep2.disabled = state.images.length === 0;
}

// Story Input
function handleStoryInput(e) {
    const value = e.target.value;
    if (value.length <= 500) {
        state.story = value;
        elements.charCount.textContent = value.length;
    } else {
        e.target.value = value.slice(0, 500);
    }
}

// Chip Selection
function handleChipSelect(e, type) {
    if (!e.target.classList.contains('chip')) return;

    const parent = e.target.parentElement;
    parent.querySelectorAll('.chip').forEach(chip => chip.classList.remove('selected'));
    e.target.classList.add('selected');

    state[type] = e.target.dataset.value;
}

// Navigation
function goToStep(step) {
    state.currentStep = step;

    // Update step indicators
    elements.stepIndicators.forEach((indicator, index) => {
        const stepNum = index + 1;
        indicator.classList.remove('active', 'completed');
        if (stepNum === step) {
            indicator.classList.add('active');
        } else if (stepNum < step) {
            indicator.classList.add('completed');
        }
    });

    // Show/hide sections
    elements.step1.classList.toggle('hidden', step !== 1);
    elements.step2.classList.toggle('hidden', step !== 2);
    elements.step3.classList.toggle('hidden', step !== 3);
}

function resetAndGoToStep2() {
    // Reset progress view
    elements.progressView.classList.remove('hidden');
    elements.resultView.classList.add('hidden');
    elements.backToStep2.classList.add('hidden');
    elements.downloadBtn.classList.add('hidden');

    // Reset progress steps
    [elements.stepAnalyze, elements.stepScript, elements.stepVideo].forEach(step => {
        step.classList.remove('active', 'completed');
        step.querySelector('.status-icon').textContent = 'hourglass_empty';
    });

    // Reset progress circle
    elements.progressCircle.style.strokeDashoffset = 283;
    elements.progressPercent.textContent = '0%';

    goToStep(2);
}

// Video Generation
async function generateVideo() {
    if (!state.story.trim()) {
        showSnackbar('스토리를 입력해주세요');
        return;
    }

    goToStep(3);

    try {
        // Prepare form data
        const formData = new FormData();
        state.images.forEach((img, index) => {
            formData.append('images', img.file);
        });
        formData.append('story', state.story);
        formData.append('style', state.style);
        formData.append('music', state.music);

        // Start generation
        const response = await fetch(`${API_BASE_URL}/generate`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('영상 생성에 실패했습니다');
        }

        const { task_id } = await response.json();

        // Poll for progress
        await pollProgress(task_id);

    } catch (error) {
        console.error('Generation error:', error);
        showSnackbar(error.message || '오류가 발생했습니다');
        goToStep(2);
    }
}

async function pollProgress(taskId) {
    const progressSteps = [
        { element: elements.stepAnalyze, start: 0, end: 30 },
        { element: elements.stepScript, start: 30, end: 50 },
        { element: elements.stepVideo, start: 50, end: 100 }
    ];

    const poll = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/progress/${taskId}`);
            const data = await response.json();

            // Update progress UI
            updateProgress(data.progress, progressSteps);

            if (data.status === 'completed') {
                showResult(data.result);
            } else if (data.status === 'failed') {
                throw new Error(data.error || '영상 생성에 실패했습니다');
            } else {
                // Continue polling
                setTimeout(poll, 1000);
            }
        } catch (error) {
            console.error('Poll error:', error);
            showSnackbar(error.message);
            goToStep(2);
        }
    };

    await poll();
}

function updateProgress(progress, steps) {
    // Update progress circle
    const offset = 283 - (283 * progress / 100);
    elements.progressCircle.style.strokeDashoffset = offset;
    elements.progressPercent.textContent = `${Math.round(progress)}%`;

    // Update step indicators
    steps.forEach((step, index) => {
        if (progress >= step.end) {
            step.element.classList.remove('active');
            step.element.classList.add('completed');
            step.element.querySelector('.status-icon').textContent = 'check_circle';
        } else if (progress >= step.start) {
            step.element.classList.add('active');
            step.element.querySelector('.status-icon').textContent = 'sync';
        }
    });
}

function showResult(result) {
    state.generatedVideo = result.video_url;
    state.generatedScript = result.script;

    // Update UI
    elements.progressView.classList.add('hidden');
    elements.resultView.classList.remove('hidden');
    elements.backToStep2.classList.remove('hidden');
    elements.downloadBtn.classList.remove('hidden');

    // Set video source
    elements.resultVideo.src = result.video_url;
    elements.generatedScript.textContent = result.script;

    showSnackbar('영상이 생성되었습니다!');
}

// Download
function downloadVideo() {
    if (!state.generatedVideo) return;

    const link = document.createElement('a');
    link.href = state.generatedVideo;
    link.download = `ganpan_shorts_${Date.now()}.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Snackbar
function showSnackbar(message) {
    const snackbar = elements.snackbar;
    snackbar.querySelector('.snackbar-text').textContent = message;
    snackbar.classList.add('show');

    setTimeout(() => {
        hideSnackbar();
    }, 4000);
}

function hideSnackbar() {
    elements.snackbar.classList.remove('show');
}

// Demo Mode (for testing without backend)
function enableDemoMode() {
    window.generateVideo = async function() {
        if (!state.story.trim()) {
            showSnackbar('스토리를 입력해주세요');
            return;
        }

        goToStep(3);

        const progressSteps = [
            { element: elements.stepAnalyze, start: 0, end: 30 },
            { element: elements.stepScript, start: 30, end: 50 },
            { element: elements.stepVideo, start: 50, end: 100 }
        ];

        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 5 + 2;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);

                // Show demo result
                setTimeout(() => {
                    showResult({
                        video_url: 'https://www.w3schools.com/html/mov_bbb.mp4',
                        script: `[AI 생성 스크립트]\n\n${state.story}\n\n이 스토리를 바탕으로 ${state.style} 스타일의 감동적인 1분 영상을 제작했습니다. ${state.music} 분위기의 배경음악과 함께 특별한 순간들을 담아냈습니다.`
                    });
                }, 500);
            }
            updateProgress(progress, progressSteps);
        }, 200);
    };

    // Override the generate button
    elements.generateBtn.onclick = window.generateVideo;
}

// Make removeImage available globally
window.removeImage = removeImage;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    init();
    // Enable demo mode for testing
    enableDemoMode();
});

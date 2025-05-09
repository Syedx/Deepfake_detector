// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize particles.js
    particlesJS('particles-js', {
        "particles": {
            "number": {
                "value": 100,
                "density": {
                    "enable": true,
                    "value_area": 800
                }
            },
            "color": {
                "value": "#3a86ff"
            },
            "shape": {
                "type": "circle",
                "stroke": {
                    "width": 0,
                    "color": "#000000"
                },
                "polygon": {
                    "nb_sides": 5
                }
            },
            "opacity": {
                "value": 0.7,
                "random": true,
                "anim": {
                    "enable": true,
                    "speed": 1,
                    "opacity_min": 0.4,
                    "sync": false
                }
            },
            "size": {
                "value": 5,
                "random": true,
                "anim": {
                    "enable": true,
                    "speed": 4,
                    "size_min": 0.3,
                    "sync": false
                }
            },
            "line_linked": {
                "enable": true,
                "distance": 150,
                "color": "#5a9bff",
                "opacity": 0.6,
                "width": 1.5
            },
            "move": {
                "enable": true,
                "speed": 3,
                "direction": "none",
                "random": true,
                "straight": false,
                "out_mode": "out",
                "bounce": false,
                "attract": {
                    "enable": true,
                    "rotateX": 600,
                    "rotateY": 1200
                }
            }
        },
        "interactivity": {
            "detect_on": "canvas",
            "events": {
                "onhover": {
                    "enable": true,
                    "mode": "grab"
                },
                "onclick": {
                    "enable": true,
                    "mode": "push"
                },
                "resize": true
            },
            "modes": {
                "grab": {
                    "distance": 180,
                    "line_linked": {
                        "opacity": 1
                    }
                },
                "bubble": {
                    "distance": 400,
                    "size": 40,
                    "duration": 2,
                    "opacity": 8,
                    "speed": 3
                },
                "repulse": {
                    "distance": 200,
                    "duration": 0.4
                },
                "push": {
                    "particles_nb": 4
                },
                "remove": {
                    "particles_nb": 2
                }
            }
        },
        "retina_detect": true
    });
    
    // Form elements
    const imageForm = document.getElementById('image-upload-form');
    const videoForm = document.getElementById('video-upload-form');
    const imageFile = document.getElementById('image-file');
    const videoFile = document.getElementById('video-file');
    const imageBrowseBtn = document.getElementById('image-browse-btn');
    const videoBrowseBtn = document.getElementById('video-browse-btn');
    const imageUploadArea = document.getElementById('image-upload-area');
    const videoUploadArea = document.getElementById('video-upload-area');
    
    // Progress steps
    const stepUpload = document.getElementById('step-upload');
    const stepAnalyze = document.getElementById('step-analyze');
    const stepResults = document.getElementById('step-results');
    
    // Results elements
    const loadingContainer = document.getElementById('loading-container');
    const resultsContainer = document.getElementById('results-container');
    const resultBadge = document.getElementById('result-badge');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const resultSummaryCard = document.getElementById('result-summary-card');
    const resultImage = document.getElementById('result-image');
    const imageResult = document.getElementById('image-result');
    const videoResult = document.getElementById('video-result');
    const resultVideo = document.getElementById('result-video');
    
    // Video stats elements
    const videoStats = document.getElementById('video-stats');
    const totalFrames = document.getElementById('total-frames');
    const realFrames = document.getElementById('real-frames');
    const fakeFrames = document.getElementById('fake-frames');
    
    // Post-processing elements
    const explanationContainer = document.getElementById('explanation-container');
    const explanationList = document.getElementById('explanation-list');
    const heatmapContainer = document.getElementById('heatmap-container');
    const heatmapImage = document.getElementById('heatmap-image');
    const basicInfoTbody = document.getElementById('basic-info-tbody');
    const exifSections = document.getElementById('exif-sections');
    
    // Loading message
    const loadingMessage = document.getElementById('loading-message');
    
    // Initialize file upload UI
    setupFileUpload();
    
    // Handle image form submission
    imageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        
        // Update progress steps
        updateProgressSteps('analyze');
        
        // Show loading
        loadingContainer.classList.remove('d-none');
        resultsContainer.classList.add('d-none');
        loadingMessage.textContent = 'Analyzing image. This may take a few moments...';
        
        // Smooth scroll to loading section
        loadingContainer.scrollIntoView({ behavior: 'smooth' });
        
        // Reset previous results
        resetResults();
        
        // Submit form data
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide loading
            loadingContainer.classList.add('d-none');
            
            if (data.error) {
                updateProgressSteps('upload');
                showError(data.error);
                return;
            }
            
            // Update progress steps
            updateProgressSteps('results');
            
            // Show results
            resultsContainer.classList.remove('d-none');
            
            // Smooth scroll to results section
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
            
            // Show image result, hide video result
            imageResult.classList.remove('d-none');
            videoResult.classList.add('d-none');
            videoStats.classList.add('d-none');
            
            // Set result image
            resultImage.src = data.result_url;
            
            // Display result badge
            displayResultBadge(data.is_real, data.confidence);
            
            // Display post-processing features
            displayExplanation(data.explanation);
            displayHeatmap(data.heatmap_url);
            displayMetadata(data.metadata);
        })
        .catch(error => {
            loadingContainer.classList.add('d-none');
            updateProgressSteps('upload');
            showError('Error processing image: ' + error.message);
            console.error('Error:', error);
        });
    });
    
    // Handle video form submission
    videoForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        
        // Update progress steps
        updateProgressSteps('analyze');
        
        // Show loading
        loadingContainer.classList.remove('d-none');
        resultsContainer.classList.add('d-none');
        loadingMessage.textContent = 'Processing video. This may take several minutes depending on the video length...';
        
        // Smooth scroll to loading section
        loadingContainer.scrollIntoView({ behavior: 'smooth' });
        
        // Reset previous results
        resetResults();
        
        // Submit form data
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide loading
            loadingContainer.classList.add('d-none');
            
            if (data.error) {
                updateProgressSteps('upload');
                showError(data.error);
                return;
            }
            
            // Update progress steps
            updateProgressSteps('results');
            
            // Show results
            resultsContainer.classList.remove('d-none');
            
            // Smooth scroll to results section
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
            
            // Show video result, hide image result
            imageResult.classList.add('d-none');
            videoResult.classList.remove('d-none');
            videoStats.classList.remove('d-none');
            
            // Set result video
            resultVideo.src = data.result_url;
            resultVideo.load();
            
            // Set video stats
            totalFrames.textContent = data.total_frames || 0;
            realFrames.textContent = data.real_frames || 0;
            fakeFrames.textContent = data.fake_frames || 0;
            
            // Display result badge
            displayResultBadge(data.is_real, data.confidence);
            
            // Hide post-processing features for videos (currently only for images)
            explanationContainer.classList.add('d-none');
            heatmapContainer.classList.add('d-none');
        })
        .catch(error => {
            loadingContainer.classList.add('d-none');
            updateProgressSteps('upload');
            showError('Error processing video: ' + error.message);
            console.error('Error:', error);
        });
    });
    
    // Function to display result badge and confidence
    function displayResultBadge(isReal, confidence) {
        // Set badge text and class
        resultBadge.textContent = isReal ? 'REAL' : 'FAKE';
        resultBadge.className = 'result-badge';
        resultBadge.classList.add(isReal ? 'badge-real' : 'badge-fake');
        
        // Set confidence bar
        confidenceBar.style.width = confidence + '%';
        confidenceBar.className = 'confidence-bar';
        confidenceBar.classList.add(isReal ? 'confidence-bar-real' : 'confidence-bar-fake');
        
        // Set confidence text
        confidenceText.textContent = confidence.toFixed(1);
        
        // Style the result card based on the result
        resultSummaryCard.className = 'result-card';
        resultSummaryCard.classList.add(isReal ? 'result-real' : 'result-fake');
    }
    
    // Function to display explanation
    function displayExplanation(explanation) {
        if (explanation && explanation.length > 0) {
            explanationContainer.classList.remove('d-none');
            explanationList.innerHTML = '';
            
            explanation.forEach(point => {
                const li = document.createElement('li');
                li.textContent = point;
                explanationList.appendChild(li);
            });
        } else {
            explanationContainer.classList.add('d-none');
        }
    }
    
    // Function to display heatmap
    function displayHeatmap(heatmapUrl) {
        if (heatmapUrl) {
            heatmapContainer.classList.remove('d-none');
            heatmapImage.src = heatmapUrl;
        } else {
            heatmapContainer.classList.add('d-none');
        }
    }
    
    // Function to display metadata
    function displayMetadata(metadata) {
        if (!metadata) return;
        
        // Display basic information
        if (metadata['Basic Information']) {
            basicInfoTbody.innerHTML = '';
            const basicInfo = metadata['Basic Information'];
            
            for (const [key, value] of Object.entries(basicInfo)) {
                const row = document.createElement('tr');
                const keyCell = document.createElement('td');
                keyCell.textContent = key;
                const valueCell = document.createElement('td');
                valueCell.textContent = value;
                
                row.appendChild(keyCell);
                row.appendChild(valueCell);
                basicInfoTbody.appendChild(row);
            }
        }
        
        // Display EXIF data
        if (metadata['EXIF Data']) {
            exifSections.innerHTML = '';
            const exifData = metadata['EXIF Data'];
            
            if (Object.keys(exifData).length === 0) {
                const noExifMsg = document.createElement('p');
                noExifMsg.textContent = 'No EXIF metadata found in this image.';
                noExifMsg.classList.add('text-muted');
                exifSections.appendChild(noExifMsg);
            } else {
                for (const [category, data] of Object.entries(exifData)) {
                    const sectionDiv = document.createElement('div');
                    sectionDiv.classList.add('mb-3');
                    
                    const categoryTitle = document.createElement('h6');
                    categoryTitle.textContent = category;
                    sectionDiv.appendChild(categoryTitle);
                    
                    const table = document.createElement('table');
                    table.classList.add('table', 'table-striped', 'table-sm');
                    
                    const tbody = document.createElement('tbody');
                    for (const [key, value] of Object.entries(data)) {
                        const row = document.createElement('tr');
                        
                        const keyCell = document.createElement('td');
                        keyCell.textContent = key;
                        keyCell.style.width = '40%';
                        
                        const valueCell = document.createElement('td');
                        valueCell.textContent = value;
                        
                        row.appendChild(keyCell);
                        row.appendChild(valueCell);
                        tbody.appendChild(row);
                    }
                    
                    table.appendChild(tbody);
                    sectionDiv.appendChild(table);
                    exifSections.appendChild(sectionDiv);
                }
            }
        }
    }
    
    // Function to setup the file upload UI
    function setupFileUpload() {
        // Image upload area click handler
        imageUploadArea.addEventListener('click', function() {
            imageFile.click();
        });
        
        // Video upload area click handler
        videoUploadArea.addEventListener('click', function() {
            videoFile.click();
        });
        
        // Image browse button click handler
        imageBrowseBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            imageFile.click();
        });
        
        // Video browse button click handler
        videoBrowseBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            videoFile.click();
        });
        
        // Image file selection handler
        imageFile.addEventListener('change', function() {
            if (this.files.length > 0) {
                updateUploadAreaWithFile(imageUploadArea, this.files[0]);
            }
        });
        
        // Video file selection handler
        videoFile.addEventListener('change', function() {
            if (this.files.length > 0) {
                updateUploadAreaWithFile(videoUploadArea, this.files[0]);
            }
        });
        
        // Drag and drop for image upload area
        setupDragAndDrop(imageUploadArea, imageFile);
        
        // Drag and drop for video upload area
        setupDragAndDrop(videoUploadArea, videoFile);
    }
    
    // Function to setup drag and drop
    function setupDragAndDrop(dropArea, fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('border-primary');
        }
        
        function unhighlight() {
            dropArea.classList.remove('border-primary');
        }
        
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                updateUploadAreaWithFile(dropArea, files[0]);
            }
        }
    }
    
    // Function to update upload area with selected file
    function updateUploadAreaWithFile(uploadArea, file) {
        const iconElement = uploadArea.querySelector('.upload-icon i');
        const textElement = uploadArea.querySelector('.upload-text');
        const subtextElement = uploadArea.querySelector('.upload-subtext');
        
        // Update the icon
        iconElement.className = 'fas fa-check-circle';
        
        // Update the text
        textElement.textContent = 'File selected';
        
        // Update the subtext
        subtextElement.textContent = `${file.name} (${formatFileSize(file.size)})`;
        
        // Add a selected class
        uploadArea.classList.add('border-success');
    }
    
    // Function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Function to update progress steps
    function updateProgressSteps(currentStep) {
        // Reset all steps
        stepUpload.classList.remove('active', 'completed');
        stepAnalyze.classList.remove('active', 'completed');
        stepResults.classList.remove('active', 'completed');
        
        // Set appropriate classes based on current step
        switch(currentStep) {
            case 'upload':
                stepUpload.classList.add('active');
                break;
            case 'analyze':
                stepUpload.classList.add('completed');
                stepAnalyze.classList.add('active');
                break;
            case 'results':
                stepUpload.classList.add('completed');
                stepAnalyze.classList.add('completed');
                stepResults.classList.add('active');
                break;
        }
    }
    
    // Function to reset results
    function resetResults() {
        resultImage.src = '';
        resultVideo.src = '';
        resultBadge.textContent = '';
        confidenceBar.style.width = '0%';
        confidenceText.textContent = '0';
        resultSummaryCard.className = 'result-card';
        totalFrames.textContent = '0';
        realFrames.textContent = '0';
        fakeFrames.textContent = '0';
        
        // Reset post-processing elements
        explanationList.innerHTML = '';
        heatmapImage.src = '';
        basicInfoTbody.innerHTML = '';
        exifSections.innerHTML = '';
    }
    
    // Function to show error message
    function showError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'alert alert-danger alert-dismissible fade show';
        errorElement.innerHTML = `
            <strong>Error:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert after the form card
        const formCard = document.querySelector('#upload-section .card');
        formCard.parentNode.insertBefore(errorElement, formCard.nextSibling);
        
        // Scroll to the error message
        errorElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Auto-dismiss after 8 seconds
        setTimeout(() => {
            errorElement.classList.remove('show');
            setTimeout(() => errorElement.remove(), 500);
        }, 8000);
    }
}); 
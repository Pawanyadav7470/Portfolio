/**
 * Pawan Yadav Portfolio Script
 * Highlights: Text Typing Effect, Interactive Skill Filtering, and a
 * client-side live Neural Network Training visualization on canvas.
 */

document.addEventListener('DOMContentLoaded', () => {
    
    /* -------------------------------------------------------------
       1. Mobile Navigation & Hamburger Menu
       ------------------------------------------------------------- */
    const menuToggle = document.getElementById('menuToggle');
    const navMenu = document.getElementById('navMenu');
    const navLinks = document.querySelectorAll('.nav-link');

    if (menuToggle && navMenu) {
        menuToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            menuToggle.classList.toggle('open');
        });

        // Close mobile menu when a nav link is clicked
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('active');
                menuToggle.classList.remove('open');
            });
        });
    }

    /* -------------------------------------------------------------
       2. Typing Text Animation (Hero Section)
       ------------------------------------------------------------- */
    const typedTextSpan = document.getElementById('typedText');
    const textArray = ["AI & ML Solutions", "Deep Learning Models", "Python Web Backends", "Intelligent Systems"];
    const typingSpeed = 100;
    const erasingSpeed = 50;
    const newTextDelay = 2000; // Delay between words
    let textArrayIndex = 0;
    let charIndex = 0;

    function type() {
        if (charIndex < textArray[textArrayIndex].length) {
            if(!typedTextSpan.classList.contains("typing")) typedTextSpan.classList.add("typing");
            typedTextSpan.textContent += textArray[textArrayIndex].charAt(charIndex);
            charIndex++;
            setTimeout(type, typingSpeed);
        } else {
            typedTextSpan.classList.remove("typing");
            setTimeout(erase, newTextDelay);
        }
    }

    function erase() {
        if (charIndex > 0) {
            if(!typedTextSpan.classList.contains("typing")) typedTextSpan.classList.add("typing");
            typedTextSpan.textContent = textArray[textArrayIndex].substring(0, charIndex - 1);
            charIndex--;
            setTimeout(erase, erasingSpeed);
        } else {
            typedTextSpan.classList.remove("typing");
            textArrayIndex++;
            if(textArrayIndex >= textArray.length) textArrayIndex = 0;
            setTimeout(type, typingSpeed + 500);
        }
    }

    if (typedTextSpan) {
        setTimeout(type, newTextDelay);
    }

    /* -------------------------------------------------------------
       3. Scrollspy Navbar Highlighting
       ------------------------------------------------------------- */
    const sections = document.querySelectorAll('section');
    
    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (pageYOffset >= (sectionTop - 150)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });

    /* -------------------------------------------------------------
       4. Interactive Technical Skills Categorization (Filters)
       ------------------------------------------------------------- */
    const filterBtns = document.querySelectorAll('.filter-btn');
    const skillCards = document.querySelectorAll('.skill-card');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Toggle active state
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const filterValue = btn.getAttribute('data-filter');

            skillCards.forEach(card => {
                const category = card.getAttribute('data-category');
                
                // Add fade out animation effect
                card.style.opacity = '0';
                card.style.transform = 'scale(0.8)';
                
                setTimeout(() => {
                    if (filterValue === 'all' || category === filterValue) {
                        card.classList.remove('hidden');
                        setTimeout(() => {
                            card.style.opacity = '1';
                            card.style.transform = 'scale(1)';
                        }, 50);
                    } else {
                        card.classList.add('hidden');
                    }
                }, 300);
            });
        });
    });


    /* -------------------------------------------------------------
       5. Client-Side Live Neural Network Training Simulation
       ------------------------------------------------------------- */
    // DOM Elements
    const datasetSelect = document.getElementById('datasetSelect');
    const hiddenNeuronsSlider = document.getElementById('hiddenNeurons');
    const neuronsValSpan = document.getElementById('neuronsVal');
    const activationSelect = document.getElementById('activationSelect');
    const trainBtn = document.getElementById('trainBtn');
    const resetPlaygroundBtn = document.getElementById('resetPlaygroundBtn');
    const epochText = document.getElementById('epochText');
    const lossText = document.getElementById('lossText');
    
    const decisionCanvas = document.getElementById('decisionCanvas');
    const networkCanvas = document.getElementById('networkCanvas');
    
    let isTraining = false;
    let trainingLoopId = null;
    let currentDataset = [];
    let lr = 0.1; // Default learning rate
    
    // Setup Learning Rate selection buttons
    const lrButtons = document.querySelectorAll('.lr-btn');
    lrButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            lrButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            lr = parseFloat(btn.getAttribute('data-value'));
        });
    });
    
    // Slider label updates
    if (hiddenNeuronsSlider && neuronsValSpan) {
        hiddenNeuronsSlider.addEventListener('input', (e) => {
            neuronsValSpan.textContent = `${e.target.value} Neurons`;
            initializePlayground();
        });
    }

    // Initialize/reset network variables
    let nn = null;

    class SimpleNeuralNetwork {
        constructor(inputSize, hiddenSize, outputSize, activationName) {
            this.inputSize = inputSize; // 2
            this.hiddenSize = hiddenSize; // adjustable
            this.outputSize = outputSize; // 1
            this.activationName = activationName;

            // Initialize weights randomly with Xavier/Glorot-like bounds
            // Adding bias term to inputs (inputSize + 1)
            this.w1 = Array.from({length: this.hiddenSize}, () => 
                Array.from({length: this.inputSize + 1}, () => (Math.random() - 0.5) * 2.0)
            );
            
            // Adding bias to hidden activations (hiddenSize + 1)
            this.w2 = Array.from({length: this.outputSize}, () => 
                Array.from({length: this.hiddenSize + 1}, () => (Math.random() - 0.5) * 2.0)
            );
            
            this.epoch = 0;
            this.loss = 0.0;
        }

        // Activations & derivatives
        activate(x) {
            if (this.activationName === 'tanh') {
                return Math.tanh(x);
            } else if (this.activationName === 'sigmoid') {
                return 1.0 / (1.0 + Math.exp(-x));
            } else { // ReLU
                return Math.max(0, x);
            }
        }

        activateDerivative(activatedVal) {
            if (this.activationName === 'tanh') {
                return 1.0 - activatedVal * activatedVal;
            } else if (this.activationName === 'sigmoid') {
                return activatedVal * (1.0 - activatedVal);
            } else { // ReLU
                return activatedVal > 0 ? 1.0 : 0.0;
            }
        }

        // Sigmoid is always used on output node for 0-1 binary probability
        sigmoid(x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        forward(inputs) {
            // inputs: [x1, x2]
            const x = [...inputs, 1.0]; // include bias element
            
            // Hidden Layer Input sum and activation
            this.h_input = [];
            this.h_activated = [];
            for (let i = 0; i < this.hiddenSize; i++) {
                let sum = 0;
                for (let j = 0; j < x.length; j++) {
                    sum += this.w1[i][j] * x[j];
                }
                this.h_input.push(sum);
                this.h_activated.push(this.activate(sum));
            }
            
            // Hidden activations with bias term
            const h = [...this.h_activated, 1.0];
            
            // Output Layer sum and activation
            this.out_input = [];
            this.out_activated = [];
            for (let i = 0; i < this.outputSize; i++) {
                let sum = 0;
                for (let j = 0; j < h.length; j++) {
                    sum += this.w2[i][j] * h[j];
                }
                this.out_input.push(sum);
                // Sigmoid boundary probabilities
                this.out_activated.push(this.sigmoid(sum));
            }
            
            return this.out_activated[0];
        }

        // Backpropagation training iteration
        trainStep(inputs, target, alpha) {
            // Forward pass
            const outputVal = this.forward(inputs);
            
            // 1. Calculate loss (MSE error gradient)
            // Error: output - target
            const error = outputVal - target;
            
            // Sigmoid output derivative: output * (1 - output)
            const d_out = error * (outputVal * (1.0 - outputVal));
            
            // Hidden activations with bias
            const h_bias = [...this.h_activated, 1.0];
            const inputs_bias = [...inputs, 1.0];
            
            // 2. Backpropagation gradients for output layer
            const dw2 = [];
            for (let i = 0; i < this.hiddenSize + 1; i++) {
                dw2.push(d_out * h_bias[i]);
            }
            
            // 3. Backpropagation gradients for hidden layer
            // Sum error contributions from output connections
            const d_hidden = [];
            for (let i = 0; i < this.hiddenSize; i++) {
                let sumError = this.w2[0][i] * d_out;
                let delta = sumError * this.activateDerivative(this.h_activated[i]);
                d_hidden.push(delta);
            }
            
            const dw1 = Array.from({length: this.hiddenSize}, () => []);
            for (let i = 0; i < this.hiddenSize; i++) {
                for (let j = 0; j < inputs_bias.length; j++) {
                    dw1[i].push(d_hidden[i] * inputs_bias[j]);
                }
            }
            
            // 4. Update Weights using gradient descent
            for (let i = 0; i < this.hiddenSize + 1; i++) {
                this.w2[0][i] -= alpha * dw2[i];
            }
            
            for (let i = 0; i < this.hiddenSize; i++) {
                for (let j = 0; j < inputs_bias.length; j++) {
                    this.w1[i][j] -= alpha * dw1[i][j];
                }
            }
            
            return error * error; // Squared error
        }
    }

    /* -------------------------------------------------------------
       6. Generate Training Datasets
       ------------------------------------------------------------- */
    function generateDataset(type) {
        const data = [];
        const numPoints = 120;
        
        if (type === 'xor') {
            for (let i = 0; i < numPoints; i++) {
                // Generate values between -1 and 1
                const x = (Math.random() - 0.5) * 2.0;
                const y = (Math.random() - 0.5) * 2.0;
                // Add noise
                const x_noise = x + (Math.random() - 0.5) * 0.1;
                const y_noise = y + (Math.random() - 0.5) * 0.1;
                // XOR condition
                const label = (x * y > 0) ? 1 : 0;
                data.push({ x: x_noise, y: y_noise, label: label });
            }
        } 
        else if (type === 'circle') {
            for (let i = 0; i < numPoints; i++) {
                const x = (Math.random() - 0.5) * 2.0;
                const y = (Math.random() - 0.5) * 2.0;
                // Radial ring classification
                const distSq = x*x + y*y;
                const label = (distSq < 0.38) ? 1 : 0;
                // Inject noise by shifting coords slightly
                data.push({ 
                    x: x + (Math.random() - 0.5) * 0.05, 
                    y: y + (Math.random() - 0.5) * 0.05, 
                    label: label 
                });
            }
        } 
        else { // linearly separable
            for (let i = 0; i < numPoints; i++) {
                const x = (Math.random() - 0.5) * 2.0;
                const y = (Math.random() - 0.5) * 2.0;
                // Separator boundary: y = -x + 0.1
                const label = (y > -x + 0.1) ? 1 : 0;
                data.push({ x: x, y: y, label: label });
            }
        }
        return data;
    }

    /* -------------------------------------------------------------
       7. Canvas Rendering Functions
       ------------------------------------------------------------- */
    
    // Draw the 2D Decision Boundary Canvas
    function drawDecisionBoundary() {
        if (!decisionCanvas) return;
        const ctx = decisionCanvas.getContext('2d');
        const width = decisionCanvas.width;
        const height = decisionCanvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // 1. Draw prediction background grid
        const resolution = 4; // grid step in pixels for performance
        for (let py = 0; py < height; py += resolution) {
            for (let px = 0; px < width; px += resolution) {
                // Map pixel coordinates [0, width] to domain [-1.2, 1.2]
                const x = ((px / width) - 0.5) * 2.4;
                const y = -((py / height) - 0.5) * 2.4; // flip y coords
                
                const prediction = nn.forward([x, y]);
                
                // Color interpolation: blue for Class 1, orange for Class 0
                // We use opacity as a metric of confidence
                let r, g, b, alpha;
                if (prediction >= 0.5) {
                    // Blue shade (var(--cyan) #06b6d4 -> rgb(6, 182, 212))
                    r = 6; g = 182; b = 212;
                    alpha = (prediction - 0.5) * 2.0; // scale to [0, 1]
                } else {
                    // Orange shade (#f59e0b -> rgb(245, 158, 11))
                    r = 245; g = 158; b = 11;
                    alpha = (0.5 - prediction) * 2.0; // scale to [0, 1]
                }
                
                ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha * 0.75})`;
                ctx.fillRect(px, py, resolution, resolution);
            }
        }
        
        // 2. Draw coordinate axes
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(width / 2, 0); ctx.lineTo(width / 2, height);
        ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2);
        ctx.stroke();

        // 3. Draw dataset data points
        currentDataset.forEach(point => {
            // Map domain coordinates back to pixels
            const px = ((point.x / 2.4) + 0.5) * width;
            const py = ((-point.y / 2.4) + 0.5) * height;
            
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, 2 * Math.PI);
            
            if (point.label === 1) {
                ctx.fillStyle = '#06b6d4'; // Cyan
                ctx.strokeStyle = '#ffffff';
            } else {
                ctx.fillStyle = '#f59e0b'; // Amber / Orange
                ctx.strokeStyle = '#ffffff';
            }
            
            ctx.lineWidth = 1.5;
            ctx.fill();
            ctx.stroke();
        });
    }

    // Draw the Feedforward Network Architecture Graph Canvas
    function drawNetworkArchitecture() {
        if (!networkCanvas) return;
        const ctx = networkCanvas.getContext('2d');
        const width = networkCanvas.width;
        const height = networkCanvas.height;
        
        ctx.clearRect(0, 0, width, height);

        // Define node layouts
        const layers = {
            input: [
                { x: width * 0.15, y: height * 0.3, label: 'x₁' },
                { x: width * 0.15, y: height * 0.7, label: 'x₂' }
            ],
            hidden: [],
            output: [
                { x: width * 0.85, y: height * 0.5, label: 'y' }
            ]
        };

        const hiddenCount = nn.hiddenSize;
        const spacing = height / (hiddenCount + 1);
        for (let i = 0; i < hiddenCount; i++) {
            layers.hidden.push({
                x: width * 0.5,
                y: spacing * (i + 1),
                label: `h${i+1}`
            });
        }

        // Draw connections / weights
        // 1. Input Layer to Hidden Layer weights
        for (let i = 0; i < layers.hidden.length; i++) {
            const hNode = layers.hidden[i];
            
            // Connected to inputs (w1 matrix dimensions: [hidden, input + 1 for bias])
            for (let j = 0; j < layers.input.length; j++) {
                const iNode = layers.input[j];
                const weight = nn.w1[i][j];
                
                ctx.beginPath();
                ctx.moveTo(iNode.x, iNode.y);
                ctx.lineTo(hNode.x, hNode.y);
                
                // Color positive weights cyan, negative red/orange
                ctx.strokeStyle = weight >= 0 ? 'rgba(6, 182, 212, 0.65)' : 'rgba(244, 63, 94, 0.65)';
                ctx.lineWidth = Math.min(Math.abs(weight) * 2.5, 6);
                ctx.stroke();
            }
        }

        // 2. Hidden Layer to Output Layer weights
        for (let i = 0; i < layers.output.length; i++) {
            const oNode = layers.output[i];
            
            for (let j = 0; j < layers.hidden.length; j++) {
                const hNode = layers.hidden[j];
                const weight = nn.w2[i][j];
                
                ctx.beginPath();
                ctx.moveTo(hNode.x, hNode.y);
                ctx.lineTo(oNode.x, oNode.y);
                
                ctx.strokeStyle = weight >= 0 ? 'rgba(6, 182, 212, 0.65)' : 'rgba(244, 63, 94, 0.65)';
                ctx.lineWidth = Math.min(Math.abs(weight) * 2.5, 6);
                ctx.stroke();
            }
        }

        // Helper to draw neural nodes (circles with labels)
        function drawNode(node, color, glowColor) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 16, 0, 2 * Math.PI);
            ctx.fillStyle = '#0f172a';
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            
            // Add subtle shadow glows to nodes
            ctx.shadowBlur = 8;
            ctx.shadowColor = glowColor;
            ctx.fill();
            ctx.stroke();
            
            // Reset shadows
            ctx.shadowBlur = 0;

            // Draw Node Text Label
            ctx.fillStyle = '#f8fafc';
            ctx.font = '11px JetBrains Mono';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.label, node.x, node.y);
        }

        // Draw input nodes (colored purple)
        layers.input.forEach(node => drawNode(node, '#8b5cf6', 'rgba(139, 92, 246, 0.4)'));
        // Draw hidden nodes (colored cyan)
        layers.hidden.forEach(node => drawNode(node, '#06b6d4', 'rgba(6, 182, 212, 0.4)'));
        // Draw output node (colored neon green)
        layers.output.forEach(node => drawNode(node, '#10b981', 'rgba(16, 185, 129, 0.4)'));
    }

    /* -------------------------------------------------------------
       8. Training Engine Loop
       ------------------------------------------------------------- */
    function initializePlayground() {
        stopTraining();
        
        const datasetType = datasetSelect.value;
        currentDataset = generateDataset(datasetType);
        
        const hiddenNeurons = parseInt(hiddenNeuronsSlider.value);
        const activation = activationSelect.value;
        
        // Initialize neural net weights
        nn = new SimpleNeuralNetwork(2, hiddenNeurons, 1, activation);
        
        epochText.textContent = '0';
        lossText.textContent = '0.000';
        
        drawDecisionBoundary();
        drawNetworkArchitecture();
    }

    function stopTraining() {
        isTraining = false;
        if (trainingLoopId) {
            cancelAnimationFrame(trainingLoopId);
            trainingLoopId = null;
        }
        trainBtn.textContent = 'Train Model';
        trainBtn.classList.remove('btn-secondary');
        trainBtn.classList.add('btn-primary');
    }

    function runTrainingLoop() {
        if (!isTraining) return;
        
        // Train 1 epoch (shuffle and perform 1 descent step for each data point)
        // Shuffle dataset to ensure stochastically sound learning
        const shuffledData = [...currentDataset].sort(() => Math.random() - 0.5);
        
        let sumSquaredError = 0.0;
        shuffledData.forEach(point => {
            const lossTerm = nn.trainStep([point.x, point.y], point.label, lr);
            sumSquaredError += lossTerm;
        });
        
        nn.epoch++;
        nn.loss = sumSquaredError / currentDataset.length;
        
        // UI Updates every few frames to prevent performance bottlenecks
        if (nn.epoch % 3 === 0 || nn.loss < 0.01) {
            epochText.textContent = nn.epoch;
            lossText.textContent = nn.loss.toFixed(4);
            
            drawDecisionBoundary();
            drawNetworkArchitecture();
        }

        // Auto stop condition when model converges or limit reached
        if (nn.loss < 0.005 || nn.epoch >= 1500) {
            stopTraining();
            // Final redraw
            epochText.textContent = nn.epoch;
            lossText.textContent = nn.loss.toFixed(4);
            drawDecisionBoundary();
            drawNetworkArchitecture();
            return;
        }
        
        trainingLoopId = requestAnimationFrame(runTrainingLoop);
    }

    // Connect control button listeners
    if (trainBtn) {
        trainBtn.addEventListener('click', () => {
            if (isTraining) {
                stopTraining();
            } else {
                isTraining = true;
                trainBtn.textContent = 'Pause Training';
                trainBtn.classList.remove('btn-primary');
                trainBtn.classList.add('btn-secondary');
                runTrainingLoop();
            }
        });
    }

    if (resetPlaygroundBtn) {
        resetPlaygroundBtn.addEventListener('click', initializePlayground);
    }

    if (datasetSelect) {
        datasetSelect.addEventListener('change', initializePlayground);
    }
    
    if (activationSelect) {
        activationSelect.addEventListener('change', initializePlayground);
    }

    /* -------------------------------------------------------------
       6. Interactive Certificate Lightbox Modal Gallery
       ------------------------------------------------------------- */
    const certCards = document.querySelectorAll('.cert-card');
    const certModal = document.getElementById('certModal');
    const modalImg = document.getElementById('modalImg');
    const modalTitle = document.getElementById('modalTitle');
    const modalIssuer = document.getElementById('modalIssuer');
    const closeModal = document.querySelector('.close-modal');
    const modalOverlay = document.querySelector('.modal-bg-overlay');
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');

    let currentCertIndex = 0;
    const certData = [];

    // Extract certificate data from HTML attributes
    certCards.forEach((card, idx) => {
        const src = card.getAttribute('data-src');
        const title = card.getAttribute('data-title');
        const issuer = card.getAttribute('data-issuer');
        
        certData.push({ src, title, issuer });

        // Bind click handler to the entire card
        card.addEventListener('click', (e) => {
            e.preventDefault();
            openCertificate(idx);
        });
    });

    function openCertificate(index) {
        if (index < 0 || index >= certData.length) return;
        currentCertIndex = index;
        updateModalContent();
        
        certModal.style.display = 'flex';
        // Allow rendering display flex before adding transition class
        setTimeout(() => {
            certModal.classList.add('active');
            certModal.setAttribute('aria-hidden', 'false');
            document.body.style.overflow = 'hidden'; // Lock background scroll
        }, 10);
    }

    function closeCertificateModal() {
        certModal.classList.remove('active');
        certModal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = ''; // Restore scroll
        setTimeout(() => {
            certModal.style.display = 'none';
        }, 300); // Match transition duration
    }

    function updateModalContent() {
        const data = certData[currentCertIndex];
        if (!data) return;
        
        // Add fade transition on image change
        modalImg.style.opacity = '0';
        modalImg.style.transform = 'scale(0.95)';
        
        setTimeout(() => {
            modalImg.src = data.src;
            modalImg.alt = data.title;
            modalTitle.textContent = data.title;
            modalIssuer.textContent = `Issued by ${data.issuer}`;
            
            modalImg.onload = () => {
                modalImg.style.opacity = '1';
                modalImg.style.transform = 'scale(1)';
            };
        }, 150);
    }

    function navigateGallery(direction) {
        let nextIndex = currentCertIndex + direction;
        if (nextIndex >= certData.length) {
            nextIndex = 0; // Loop back to start
        } else if (nextIndex < 0) {
            nextIndex = certData.length - 1; // Loop to end
        }
        currentCertIndex = nextIndex;
        updateModalContent();
    }

    // Modal Action Listeners
    if (closeModal) {
        closeModal.addEventListener('click', closeCertificateModal);
    }
    if (modalOverlay) {
        modalOverlay.addEventListener('click', closeCertificateModal);
    }
    if (prevBtn) {
        prevBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            navigateGallery(-1);
        });
    }
    if (nextBtn) {
        nextBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            navigateGallery(1);
        });
    }

    // Keyboard Controls
    document.addEventListener('keydown', (e) => {
        if (!certModal || !certModal.classList.contains('active')) return;
        
        if (e.key === 'Escape') {
            closeCertificateModal();
        } else if (e.key === 'ArrowRight') {
            navigateGallery(1);
        } else if (e.key === 'ArrowLeft') {
            navigateGallery(-1);
        }
    });

    // Trigger initial renders of playground
    initializePlayground();
});

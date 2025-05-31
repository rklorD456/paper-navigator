/**
 * @typedef {Object} Paper
 * @property {string} id - Unique identifier for the paper
 * @property {string} title - Paper title
 * @property {string} authors - Paper authors
 * @property {string} abstract - Paper abstract
 * @property {string} url - URL to the full paper
 * @property {number} citation_count - Number of citations
 * @property {string} year - Publication year
 */

// State management
const state = {
    currentPaper: null,
    isLoading: false,
    error: null,
    retryCount: 0,
    maxRetries: 3
};

// DOM Elements
const elements = {
    paperContainer: document.getElementById('paper-container'),
    paperContent: document.getElementById('paper-content'),
    loadingIndicator: document.getElementById('loading'),
    noPapersMessage: document.getElementById('no-papers'),
    feedbackMessage: document.getElementById('feedback-message'),
    errorContainer: document.getElementById('error-container'),
    errorMessage: document.getElementById('error-message'),
    retryButton: document.getElementById('retry-button'),
    rateHelpfulBtn: document.getElementById('rate-helpful'),
    rateNotRelevantBtn: document.getElementById('rate-not-relevant'),
    skipPaperBtn: document.getElementById('skip-paper')
};

// Common utility functions
const utils = {
    // Show a global error message
    showError: function(message) {
        const errorDiv = document.getElementById('global-error');
        if (errorDiv) {
            errorDiv.querySelector('span').textContent = message;
            errorDiv.classList.remove('hidden');
            setTimeout(() => {
                errorDiv.classList.add('hidden');
            }, 5000);
        }
    },

    // Show a success message
    showSuccess: function(message) {
        const successDiv = document.getElementById('global-success');
        if (successDiv) {
            successDiv.querySelector('span').textContent = message;
            successDiv.classList.remove('hidden');
            setTimeout(() => {
                successDiv.classList.add('hidden');
            }, 3000);
        }
    },

    // Handle API errors
    handleApiError: function(error) {
        console.error('API Error:', error);
        let errorMessage = 'An error occurred. Please try again.';
        
        if (error.response) {
            try {
                const data = error.response.json();
                errorMessage = data.detail || errorMessage;
            } catch (e) {
                errorMessage = error.response.statusText || errorMessage;
            }
        } else if (error.message) {
            errorMessage = error.message;
        }
        
        this.showError(errorMessage);
        return errorMessage;
    },

    // Make an API request with error handling
    async apiRequest: async function(url, options = {}) {
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    ...options.headers
                }
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Request failed');
            }

            return data;
        } catch (error) {
            throw this.handleApiError(error);
        }
    },

    showLoading() {
        elements.loadingIndicator.classList.remove('hidden');
        elements.paperContainer.classList.add('hidden');
        elements.errorContainer.classList.add('hidden');
        state.isLoading = true;
    },

    hideLoading() {
        elements.loadingIndicator.classList.add('hidden');
        elements.paperContainer.classList.remove('hidden');
        state.isLoading = false;
    },

    showFeedback(message = 'Rating saved!', type = 'success') {
        elements.feedbackMessage.textContent = message;
        elements.feedbackMessage.className = `feedback-message feedback-${type} animate-slide-in`;
        elements.feedbackMessage.classList.remove('hidden');
        setTimeout(() => {
            elements.feedbackMessage.classList.add('hidden');
        }, 2000);
    },

    showNoPapers() {
        elements.paperContainer.classList.add('hidden');
        elements.noPapersMessage.classList.remove('hidden');
    },

    validatePaper(paper) {
        const requiredFields = ['id', 'title', 'authors', 'abstract', 'url'];
        return requiredFields.every(field => paper && paper[field]);
    }
};

// API Functions
const api = {
    async fetchRecommendation() {
        try {
            utils.showLoading();
            const response = await fetch('/api/recommendations');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data || Object.keys(data).length === 0) {
                utils.showNoPapers();
                return null;
            }
            
            if (!utils.validatePaper(data)) {
                throw new Error('Invalid paper data received from server');
            }
            
            state.currentPaper = data;
            state.retryCount = 0;
            return data;
        } catch (error) {
            console.error('Error fetching recommendation:', error);
            if (state.retryCount < state.maxRetries) {
                state.retryCount++;
                return await this.fetchRecommendation();
            }
            utils.showError('Failed to fetch paper recommendation. Please try again.');
            return null;
        } finally {
            utils.hideLoading();
        }
    },

    async submitRating(rating) {
        if (!state.currentPaper) return;

        try {
            utils.showLoading();
            const formData = new URLSearchParams();
            formData.append('paper_id', state.currentPaper.id);
            formData.append('rating', rating);

            const response = await fetch('/api/rate-paper', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.status === 'success') {
                utils.showFeedback('Rating submitted successfully!');
                await this.fetchRecommendation();
            } else {
                throw new Error('Failed to submit rating');
            }
        } catch (error) {
            console.error('Error submitting rating:', error);
            utils.showError('Failed to submit rating. Please try again.');
        } finally {
            utils.hideLoading();
        }
    }
};

// Display Functions
const display = {
    paper(paper) {
        if (!paper) return;
        
        const citationCount = paper.citation_count || 0;
        const year = paper.year || 'N/A';
        
        elements.paperContent.innerHTML = `
            <h2 class="paper-title">${paper.title}</h2>
            <div class="paper-meta">
                <span class="paper-meta-item">
                    <i class="fas fa-users"></i>
                    ${paper.authors}
                </span>
                <span class="paper-meta-item">
                    <i class="fas fa-calendar"></i>
                    ${year}
                </span>
                <span class="paper-meta-item">
                    <i class="fas fa-quote-right"></i>
                    ${citationCount} citations
                </span>
            </div>
            <div class="paper-abstract">
                <h3 class="text-lg font-semibold mb-2">Abstract</h3>
                <p>${paper.abstract}</p>
            </div>
            <a href="${paper.url}" target="_blank" class="btn btn-primary">
                <i class="fas fa-external-link-alt"></i>
                Read Full Paper
            </a>
        `;
    }
};

// Event Handlers
const handlers = {
    async handleRating(rating) {
        if (state.isLoading) return;
        await api.submitRating(rating);
    },

    async handleRetry() {
        utils.hideError();
        await api.fetchRecommendation();
    },

    handleKeyPress(event) {
        if (state.isLoading) return;
        
        switch(event.key) {
            case '1':
                handlers.handleRating(1);
                break;
            case '2':
                handlers.handleRating(-1);
                break;
            case '3':
                api.fetchRecommendation();
                break;
        }
    }
};

// Event Listeners
function initializeEventListeners() {
    elements.rateHelpfulBtn.addEventListener('click', () => handlers.handleRating(1));
    elements.rateNotRelevantBtn.addEventListener('click', () => handlers.handleRating(-1));
    elements.skipPaperBtn.addEventListener('click', () => api.fetchRecommendation());
    elements.retryButton.addEventListener('click', handlers.handleRetry);
    document.addEventListener('keypress', handlers.handleKeyPress);
}

// Initialize application
function initializeApp() {
    initializeEventListeners();
    api.fetchRecommendation().then(paper => {
        if (paper) {
            display.paper(paper);
        }
    });
}

// Start the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add global error handler
    window.showError = utils.showError;
    window.showSuccess = utils.showSuccess;
    window.handleApiError = utils.handleApiError;
    window.apiRequest = utils.apiRequest;

    initializeApp();
}); 
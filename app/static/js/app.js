/**
 * Food Image Retrieval System - Frontend JavaScript
 * Handles search functionality, result display, and user interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const searchBtn = document.getElementById('search-btn');
    const ingredientInput = document.getElementById('ingredient-input');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');
    const resultsContainer = document.getElementById('results-container');
    const resultsGrid = document.getElementById('results-grid');
    const sortSelect = document.getElementById('sort-by');
    const exampleTags = document.querySelectorAll('.example-tag');
    
    // Modal Elements
    const imageModal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');
    const modalTitle = document.getElementById('modal-title');
    const modalIngredients = document.getElementById('modal-ingredients');
    const modalScore = document.getElementById('modal-score');
    const closeModal = document.querySelector('.close-modal');
    const generateBtn = document.getElementById('generate-btn');
    const compareBtn = document.getElementById('compare-btn');
    const stars = document.querySelectorAll('.star');
    
    // State
    let currentQuery = '';
    let currentResults = [];
    let selectedImageId = null;
    let userRating = 0;
    
    // Event Listeners
    searchBtn.addEventListener('click', performSearch);
    ingredientInput.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') performSearch();
    });
    sortSelect.addEventListener('change', sortResults);
    closeModal.addEventListener('click', () => {
        imageModal.classList.add('hidden');
    });
    
    // Example tag click handling
    exampleTags.forEach(tag => {
        tag.addEventListener('click', () => {
            const ingredients = tag.dataset.ingredients;
            ingredientInput.value = ingredients;
            performSearch();
        });
    });
    
    // Star rating system
    stars.forEach(star => {
        star.addEventListener('click', () => {
            const rating = parseInt(star.dataset.rating);
            userRating = rating;
            
            // Visual feedback
            stars.forEach(s => {
                const starRating = parseInt(s.dataset.rating);
                if (starRating <= rating) {
                    s.classList.add('active');
                } else {
                    s.classList.remove('active');
                }
            });
            
            // Submit feedback if we have an active image
            if (selectedImageId) {
                submitFeedback(selectedImageId, rating);
            }
        });
    });
    
    // Modal buttons
    generateBtn.addEventListener('click', () => {
        if (selectedImageId) {
            generateVariant(selectedImageId);
        }
    });
    
    compareBtn.addEventListener('click', () => {
        if (selectedImageId) {
            // Show a UI to select another image to compare with
            alert('Compare functionality coming soon! Select another image to compare with.');
        }
    });
    
    // Handle clicks outside the modal to close it
    window.addEventListener('click', (e) => {
        if (e.target === imageModal) {
            imageModal.classList.add('hidden');
        }
    });
    
    /**
     * Perform ingredient search
     */
    async function performSearch() {
        const ingredients = ingredientInput.value.trim();
        if (!ingredients) {
            showError('Please enter ingredients to search for');
            return;
        }
        
        currentQuery = ingredients;
        showLoading(true);
        clearResults();
        clearError();
        
        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ingredients: ingredients,
                    limit: 20
                }),
            });
            
            const data = await response.json();
            
            if (data.success) {
                currentResults = data.results;
                displayResults(data.results);
                resultsContainer.classList.remove('hidden');
            } else {
                showError(data.error || 'An error occurred while searching');
            }
        } catch (error) {
            showError('Network error: ' + error.message);
        } finally {
            showLoading(false);
        }
    }
    
    /**
     * Display search results in the grid
     */
    function displayResults(results) {
        resultsGrid.innerHTML = '';
        
        if (results.length === 0) {
            resultsGrid.innerHTML = '<p class="no-results">No results found. Try different ingredients.</p>';
            return;
        }
        
        results.forEach(result => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.dataset.id = result.id;
            
            // Create card content
            card.innerHTML = `
                <img src="data:image/png;base64,${result.image}" alt="Food" class="result-image">
                <div class="result-details">
                    <div class="result-title">${result.id}</div>
                    <div class="result-ingredients">${result.matched_ingredients.join(', ')}</div>
                    <div class="result-score">Score: ${result.score}</div>
                </div>
            `;
            
            // Add click handler to open modal
            card.addEventListener('click', () => {
                openImageDetails(result);
            });
            
            resultsGrid.appendChild(card);
        });
    }
    
    /**
     * Sort results based on selected criteria
     */
    function sortResults() {
        const sortType = sortSelect.value;
        
        if (currentResults.length === 0) return;
        
        switch (sortType) {
            case 'score':
                currentResults.sort((a, b) => b.score - a.score);
                break;
            case 'ingredients':
                currentResults.sort((a, b) => {
                    const aCount = a.matched_ingredients ? a.matched_ingredients.length : 0;
                    const bCount = b.matched_ingredients ? b.matched_ingredients.length : 0;
                    return bCount - aCount;
                });
                break;
            case 'diversity':
                // This would require more complex logic for diversity
                // For now, just shuffle to show different results
                shuffleArray(currentResults);
                break;
        }
        
        displayResults(currentResults);
    }
    
    /**
     * Open image details modal
     */
    function openImageDetails(result) {
        selectedImageId = result.id;
        modalImage.src = `data:image/png;base64,${result.image}`;
        modalTitle.textContent = result.id;
        modalScore.textContent = `${result.score.toFixed(4)}`;
        
        // Clear previous ingredients
        modalIngredients.innerHTML = '';
        
        // Add matched ingredients
        if (result.matched_ingredients && result.matched_ingredients.length > 0) {
            result.matched_ingredients.forEach(ingredient => {
                const li = document.createElement('li');
                li.textContent = ingredient;
                modalIngredients.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No specific ingredients matched';
            modalIngredients.appendChild(li);
        }
        
        // Reset star ratings
        stars.forEach(star => star.classList.remove('active'));
        userRating = 0;
        
        // Show modal
        imageModal.classList.remove('hidden');
    }
    
    /**
     * Submit feedback for an image
     */
    async function submitFeedback(imageId, rating) {
        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: currentQuery,
                    image_id: imageId,
                    rating: rating
                }),
            });
            
            const data = await response.json();
            if (data.success) {
                console.log('Feedback submitted successfully');
            } else {
                console.error('Error submitting feedback:', data.error);
            }
        } catch (error) {
            console.error('Network error submitting feedback:', error.message);
        }
    }
    
    /**
     * Generate a variant of the image
     */
    async function generateVariant(imageId) {
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_id: imageId,
                    add_ingredients: [],
                    remove_ingredients: []
                }),
            });
            
            const data = await response.json();
            if (data.success) {
                alert('Image generation requested. This feature is coming soon!');
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('Network error: ' + error.message);
        }
    }
    
    // Utility functions
    function showLoading(isLoading) {
        if (isLoading) {
            loadingIndicator.classList.remove('hidden');
        } else {
            loadingIndicator.classList.add('hidden');
        }
    }
    
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }
    
    function clearError() {
        errorMessage.textContent = '';
        errorMessage.classList.add('hidden');
    }
    
    function clearResults() {
        resultsGrid.innerHTML = '';
        currentResults = [];
    }
    
    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
}); 
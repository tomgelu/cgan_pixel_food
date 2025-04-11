// Debug script to fix modal issues
document.addEventListener('DOMContentLoaded', function() {
    console.log('Debug script loaded');
    
    // Make modal globally accessible for debugging
    window.showModal = function() {
        const modal = document.getElementById('image-modal');
        if (modal) {
            console.log('Showing modal');
            // Override all styles directly
            modal.setAttribute('style', 'display: flex !important; position: fixed !important; top: 0 !important; left: 0 !important; width: 100% !important; height: 100% !important; background-color: rgba(0, 0, 0, 0.7) !important; z-index: 1000 !important;');
            return true;
        } else {
            console.error('Modal not found in DOM');
            return false;
        }
    };
    
    // Direct click handlers for all result cards
    function addCardClickHandlers() {
        console.log('Adding click handlers to cards');
        const cards = document.querySelectorAll('.result-card');
        cards.forEach(card => {
            card.onclick = function() {
                console.log('Card clicked:', card.dataset.id);
                window.showModal();
            };
        });
        console.log(`Added handlers to ${cards.length} cards`);
    }
    
    // Watch for dynamically added results
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                console.log('DOM changed, checking for cards');
                setTimeout(addCardClickHandlers, 100);
            }
        });
    });
    
    // Start observing the results container
    const resultsContainer = document.getElementById('results');
    if (resultsContainer) {
        observer.observe(resultsContainer, { childList: true, subtree: true });
        console.log('Observing results container for changes');
    } else {
        console.error('Results container not found');
    }
    
    // Add a debug button to test modal
    const container = document.querySelector('.container');
    if (container) {
        const debugButton = document.createElement('button');
        debugButton.textContent = 'Debug: Test Modal';
        debugButton.style.marginTop = '20px';
        debugButton.style.backgroundColor = '#ff5722';
        debugButton.style.color = 'white';
        debugButton.style.border = 'none';
        debugButton.style.padding = '10px 20px';
        debugButton.style.borderRadius = '4px';
        debugButton.style.cursor = 'pointer';
        
        debugButton.onclick = function() {
            window.showModal();
        };
        
        container.appendChild(debugButton);
        console.log('Debug button added');
    }
    
    // Initial check for existing cards
    addCardClickHandlers();
}); 
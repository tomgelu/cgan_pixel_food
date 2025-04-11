#!/usr/bin/env python3
"""
Routes for the Food Image Retrieval web application.
Defines all web endpoints and connects the frontend with the core functionality.
"""

import os
import json
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, current_app, send_from_directory

from core.clip_retriever import CLIPRetriever
from core.ingredient_mapper import IngredientMapper
from utils import RetrieverRegistry

def configure_routes(app):
    """Configure all routes for the Flask application."""
    
    # Initialize core components
    ingredient_mapper = IngredientMapper()
    # Get the clip_retriever from the app context
    clip_retriever = app.clip_retriever
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/search', methods=['POST'])
    def search():
        """Handle ingredient-based search requests using CLIP model only."""
        data = request.get_json()
        query = data.get('ingredients', '')
        top_k = int(data.get('limit', 10))
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Please enter ingredients to search for'
            })
        
        try:
            # Parse ingredients using create_search_query method
            canonical_ingredients, enhanced_prompt = ingredient_mapper.create_search_query(query)
            
            # Get search results directly from CLIP model (no ranking)
            results = clip_retriever.retrieve(
                query=enhanced_prompt,
                ingredients=canonical_ingredients,
                top_k=top_k
            )
            
            # Process results for display
            formatted_results = []
            for result in results:
                try:
                    img_path = result['image_path']
                    
                    # Load image and convert to base64 (no processing)
                    img = Image.open(img_path).convert('RGB')
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Just get the basic file information
                    formatted_results.append({
                        'id': result.get('id', os.path.basename(img_path)),
                        'score': round(result.get('score', 0), 4),
                        'matched_ingredients': result.get('matched_ingredients', []),
                        'image': img_base64,
                        'metadata': result.get('metadata', {})
                    })
                except Exception as e:
                    current_app.logger.error(f"Error processing result {result}: {e}")
            
            return jsonify({
                'success': True,
                'query': query,
                'enhanced_prompt': enhanced_prompt,
                'canonical_ingredients': canonical_ingredients,
                'results': formatted_results
            })
        
        except Exception as e:
            current_app.logger.error(f"Search error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'An error occurred: {str(e)}'
            })
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files."""
        return send_from_directory(app.static_folder, filename)
    
    return app 
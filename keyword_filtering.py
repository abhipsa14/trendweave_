def filter_text(self, text: str, mode: FilterMode = FilterMode.INCLUDE,
                use_semantic: bool = True, 
                similarity_method: SimilarityMethod = SimilarityMethod.COSINE,
                auto_extract_keywords: bool = False,
                **kwargs) -> FilterResult:
    """
    Filter text using KeyBERT and semantic similarity
    
    Args:
        text: Input text to filter
        mode: Filtering mode
        use_semantic: Whether to use semantic matching
        similarity_method: Method for calculating similarity
        auto_extract_keywords: Whether to auto-extract keywords from text
    """
    
    # Auto-extract keywords if requested
    extracted_keywords = []
    if auto_extract_keywords:
        extracted_keywords = self.extract_keywords_from_text(text, **kwargs)
        # Add extracted keywords to filter
        auto_keywords = [kw for kw, _ in extracted_keywords[:5]]  # Top 5
        self.add_keywords(auto_keywords)
    
    # Define the pattern for exact matches
    pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in self.keywords) + r')\b'
    
    # Find matches
    exact_matches = self.find_exact_matches(text, pattern)  # Pass the pattern here
    semantic_matches = []
    
    if use_semantic:
        semantic_matches = self.find_semantic_matches(text, similarity_method)
    
    # Combine and deduplicate matches
    all_matches = exact_matches + semantic_matches
    all_matches = self._deduplicate_matches(all_matches)
    
    # Calculate relevance score
    relevance_score = self._calculate_relevance_score(text, all_matches)
    
    # Apply filtering based on mode
    filtered_text = self._apply_filtering(text, all_matches, mode, **kwargs)
    
    return FilterResult(
        original_text=text,
        filtered_text=filtered_text,
        matches=all_matches,
        extracted_keywords=extracted_keywords,
        semantic_matches=[self._match_to_dict(m) for m in semantic_matches],
        relevance_score=relevance_score,
        match_count=len(all_matches)
    )

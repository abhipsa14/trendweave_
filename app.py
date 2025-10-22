# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tempfile
import os
import base64
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="TrendWeave",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #667eea;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #f093fb, #f5576c);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .accessory-card {
        background: linear-gradient(45deg, #4facfe, #00f2fe);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .trend-card {
        background: linear-gradient(45deg, #43e97b, #38f9d7);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FashionTrendApp:
    def __init__(self):
        try:
            from fashion_processor import FashionProcessor
            from fashion_forecaster import FashionTrendForecaster
            from pdf_extractors import PDFProcessor
            from data_manager import FashionDataManager
            from model_trainer import FashionModelTrainer
            
            self.processor = FashionProcessor()
            self.forecaster = FashionTrendForecaster()
            self.pdf_processor = PDFProcessor()
            self.data_manager = FashionDataManager()
            self.model_trainer = FashionModelTrainer()
            
        except ImportError as e:
            st.error(f"Failed to import required modules: {e}")
            st.stop()
    
    def run(self):
        # Header
        st.markdown('<h1 class="main-header">TrendWeave</h1>', unsafe_allow_html=True)
        st.write("Fine-tune models and generate comprehensive fashion forecasts including accessories")
        
        # Sidebar
        st.sidebar.title("AI Models")
        
        # Show model status
        model_status = self.model_trainer.get_training_status()
        fine_tuned_count = sum(model_status.values())
        
        if fine_tuned_count >= 3:
            st.sidebar.success(f"✅ {fine_tuned_count}/4 Fine-tuned")
        else:
            st.sidebar.info(f"🔄 {fine_tuned_count}/4 Fine-tuned")
        
        # Navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Analyze PDFs", "Fine-tune Models", "Accessories Dashboard"])
        
        if page == "Analyze PDFs":
            self.show_analysis_interface()
        elif page == "Fine-tune Models":
            self.show_training_interface()
        else:
            self.show_accessories_dashboard()
    
    def show_analysis_interface(self):
        """Show PDF analysis interface"""
        st.sidebar.title("📁 Upload PDFs")
        
        # Add minimum file requirement info
        st.sidebar.info("Upload at least 3 PDFs for comprehensive analysis")
        
        uploaded_files = st.sidebar.file_uploader(
            "Upload fashion PDF documents (Minimum 3 required)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select 3 or more fashion PDFs for trend analysis"
        )
        
        if uploaded_files:
            self.process_uploaded_files(uploaded_files)
        else:
            self.show_welcome_screen()
    
    def validate_uploaded_files(self, uploaded_files):
        """Validate that at least 3 PDFs are uploaded"""
        if len(uploaded_files) < 3:
            st.error(f"❌ Please upload at least 3 PDFs. Currently uploaded: {len(uploaded_files)}")
            st.info("""
            **💡 Why 3+ documents?**
            - Better trend detection across multiple sources
            - More accurate forecasting
            - Comprehensive market analysis
            - Richer training data for model fine-tuning
            """)
            return False
        
        # Additional validation for file types
        non_pdf_files = [f.name for f in uploaded_files if not f.name.lower().endswith('.pdf')]
        if non_pdf_files:
            st.error(f"Non-PDF files detected: {', '.join(non_pdf_files)}")
            return False
            
        return True
    
    def show_welcome_screen(self):
        """Show welcome screen"""
        st.info("""
        ## 🚀 Welcome to TrendWeave
        
        **How to use:**
        1. **Upload PDFs** - Upload at least 3 fashion documents for comprehensive analysis
        2. **Fine-tune Models** - Train on your data for better accuracy  
        3. **Generate Forecasts** - Get 12-month trend predictions including accessories
        
        **📋 Requirements:**
        - Minimum 3 PDF documents
        - Fashion-related content (reports, catalogs, trend analysis)
        - Supported formats: PDF
        
        **Why 3+ documents?**
        - Better trend detection across multiple sources
        - More accurate forecasting
        - Richer training data for model fine-tuning
        """)
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF files with minimum requirement"""
        
        # Validate minimum file requirement
        if not self.validate_uploaded_files(uploaded_files):
            return
        
        # Show file count confirmation
        st.success(f"✅ {len(uploaded_files)} PDFs uploaded for analysis")
        
        # Show file list
        with st.expander("📄 Uploaded Files", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                file_size = len(file.getvalue()) / 1024  # Size in KB
                st.write(f"{i}. {file.name} ({file_size:.1f} KB)")
        
        # Show analysis progress
        st.subheader("Analysis Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_analyses = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Extract and analyze text
                text = self.pdf_processor.extract_text_from_pdf(tmp_path)
                
                if text and self.pdf_processor.is_fashion_related(text):
                    analysis = self.processor.analyze_fashion_trends(text)
                    analysis['filename'] = uploaded_file.name
                    analysis['text_preview'] = text[:500] + "..." if len(text) > 500 else text
                    all_analyses.append(analysis)
                    
                    # Display individual results
                    self.display_analysis_results(analysis)
                else:
                    st.warning(f"⚠️ {uploaded_file.name} - Limited fashion content found")
                
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Show analysis summary
        if all_analyses:
            successful_analyses = len(all_analyses)
            st.success(f"🎉 Successfully analyzed {successful_analyses}/{len(uploaded_files)} documents")
            
            # Collect training data
            training_data = self.data_manager.collect_training_data_from_analysis(all_analyses)
            if training_data:
                self.data_manager.save_training_data(training_data)
                st.sidebar.success(f"📚 Collected {len(training_data)} training samples")
            
            # Generate forecast only if we have at least 3 successful analyses
            if successful_analyses >= 3:
                self.generate_and_display_forecast(all_analyses)
            else:
                st.warning("Need at least 3 successfully analyzed documents for forecasting")
    
    def display_analysis_results(self, analysis):
        """Display individual analysis results"""
        st.markdown(f"### 📄 {analysis['filename']}")
        
        # Sentiment
        if 'sentiment_analysis' in analysis:
            sentiment = analysis['sentiment_analysis']
            if 'error' not in sentiment:
                col1, col2, col3 = st.columns(3)
                with col1:
                    icon = "😊" if sentiment['overall_sentiment'] == 'POSITIVE' else "😐" if sentiment['overall_sentiment'] == 'NEUTRAL' else "😞"
                    st.metric("Sentiment", f"{icon} {sentiment['overall_sentiment']}")
                with col2:
                    st.metric("Confidence", f"{sentiment.get('confidence', 0):.2f}")
                with col3:
                    st.metric("Document Type", "Fashion Content")
        
        # Keywords
        if 'keyword_extraction' in analysis:
            keywords = analysis['keyword_extraction']
            if 'keywords' in keywords and 'error' not in keywords:
                st.write("**🔑 Key Keywords:**")
                keyword_cols = st.columns(4)
                for idx, (keyword, score) in enumerate(keywords['keywords'][:8]):
                    with keyword_cols[idx % 4]:
                        st.markdown(f'<div class="metric-card">{keyword}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
    
    def generate_and_display_forecast(self, analyses):
        """Generate and display comprehensive forecast including accessories"""
        st.markdown('<h2 class="section-header">🎯12-Month Fashion Forecast</h2>', unsafe_allow_html=True)
        
        with st.spinner("Generating comprehensive fashion forecast..."):
            forecast = self.forecaster.generate_forecast_report(analyses, len(analyses))
        
        # Executive Summary
        st.markdown('<h3 class="section-header">Executive Summary</h3>', unsafe_allow_html=True)
        summary = forecast['executive_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Market Outlook", summary['market_outlook'])
        with col2:
            st.metric("Consumer Sentiment", summary['consumer_sentiment'])
        with col3:
            st.metric("Forecast Confidence", summary['forecast_confidence'])
        with col4:
            st.metric("Accessories Growth", summary.get('accessories_growth_outlook', 'High'))
        
        # Trend Forecasts
        st.markdown('<h3>Main Fashion Trends</h3>', unsafe_allow_html=True)
        trends = forecast['trend_forecasts']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🛍️ Product Trends:**")
            for trend in trends['product_trends']:
                st.write(f"- {trend}")

            
            st.write("**🎨 Color Trends:**")
            for trend in trends['color_trends']:
                st.write(f"- {trend}")

        
        with col2:
            st.write("**🧵 Material Trends:**")
            for trend in trends['material_trends']:
                st.write(f"- {trend}")
          
            st.write("**🌟 Style Trends:**")
            for trend in trends['style_trends']:
                st.write(f"- {trend}")
        
        # Accessories Forecast
        self.display_accessories_forecast(forecast['accessories_forecast'])
        
        # Seasonal Outlook
        self.display_seasonal_outlook(forecast['seasonal_outlook'])
        
        # Market Recommendations
        self.display_market_recommendations(forecast['market_recommendations'])
        
        # Generate report
        self.generate_report(forecast, analyses)
    
    def display_accessories_forecast(self, accessories_forecast):
        """Display accessories forecast section"""
        st.markdown('<h3 class="section-header">Accessories Forecasting</h3>', unsafe_allow_html=True)
        
        # Key Categories
        st.subheader("Accessory Categories")
        categories = accessories_forecast['key_accessory_categories']
        
        for category, data in categories.items():
            with st.expander(f"📦 {category.title()}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Trend", data['trend'])
                    st.metric("Growth Outlook", data['growth_outlook'])
                with col2:
                    st.write("**Key Items:**")
                    for item in data['key_items']:
                        st.write(f"{item}")
                    st.write(f"**Consumer Demand:** {data['consumer_demand']}")
        
        # Materials Trends
        st.subheader("Materials Trends")
        materials = accessories_forecast['materials_trends']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Sustainable Materials**")
            for material in materials['sustainable_materials']:
                st.write(f"- {material}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.write("**Luxury Materials**")
            for material in materials['luxury_materials']:
                st.write(f"- {material}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.write("**Innovative Materials**")
            for material in materials['innovative_materials']:
                st.write(f"- {material}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Growth Visualization
        st.subheader("📈 Category Growth Forecast")
        growth_data = accessories_forecast['category_growth_forecast']
        
        # Create growth chart
        growth_rates = growth_data['projected_growth_rates']
        fig = px.bar(
            x=list(growth_rates.keys()),
            y=[float(rate.split('-')[0].strip('%')) for rate in growth_rates.values()],
            title="Projected Growth Rates by Category",
            labels={'x': 'Category', 'y': 'Growth Rate (%)'},
            color=[float(rate.split('-')[0].strip('%')) for rate in growth_rates.values()],
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_seasonal_outlook(self, seasonal_outlook):
        """Display seasonal outlook"""
        st.markdown('<h3 class="section-header">Seasonal Outlook</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spring-Summer")
            spring = seasonal_outlook['Spring-Summer']
            st.write(f"**Key Themes:** {', '.join(spring['key_themes'])}")
            st.write(f"**Color Story:** {spring['color_story']}")
            st.write(f"**Fabric Focus:** {spring['fabric_focus']}")
            st.write(f"**Accessory Focus:** {spring['accessory_focus']}")
        
        with col2:
            st.subheader("Fall-Winter")
            fall = seasonal_outlook['Fall-Winter']
            st.write(f"**Key Themes:** {', '.join(fall['key_themes'])}")
            st.write(f"**Color Story:** {fall['color_story']}")
            st.write(f"**Fabric Focus:** {fall['fabric_focus']}")
            st.write(f"**Accessory Focus:** {fall['accessory_focus']}")
    
    def display_market_recommendations(self, recommendations):
        """Display market recommendations"""
        st.markdown('<h3 class="section-header">Market Recommendations</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Product Development")
            for rec in recommendations['product_development']:
                st.write(f"• {rec}")
        
        with col2:
            st.subheader("Marketing Strategy")
            for rec in recommendations['marketing_strategy']:
                st.write(f"• {rec}")
        
        with col3:
            st.subheader("Retail Innovation")
            for rec in recommendations['retail_innovation']:
                st.write(f"• {rec}")
        
        # Accessory-specific recommendations
        if 'accessory_strategy' in recommendations:
            st.subheader("Accessory Strategy")
            for rec in recommendations['accessory_strategy']:
                st.write(f"• {rec}")
    
    def generate_report(self, forecast, analyses):
        """Generate downloadable report"""
        st.markdown("### Download Forecast Report")
        
        if st.button("Generate Report"):
            report_content = self.create_comprehensive_report(forecast, analyses)
            
            b64 = base64.b64encode(report_content.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="fashion_forecast_{forecast["report_metadata"]["report_id"]}.txt">📥 Download Forecast Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def create_comprehensive_report(self, forecast, analyses):
        """Create comprehensive report content including accessories"""
        report = f"""FASHION TREND FORECAST REPORT
Generated: {forecast['report_metadata']['generation_date']}
Period: {forecast['report_metadata']['forecast_period']}
Documents Analyzed: {forecast['report_metadata']['documents_analyzed']}
Report ID: {forecast['report_metadata']['report_id']}

EXECUTIVE SUMMARY
Market Outlook: {forecast['executive_summary']['market_outlook']}
Consumer Sentiment: {forecast['executive_summary']['consumer_sentiment']}
Forecast Confidence: {forecast['executive_summary']['forecast_confidence']}
Accessories Growth: {forecast['executive_summary'].get('accessories_growth_outlook', 'High')}

TREND FORECASTS
"""
        
        for category, trends in forecast['trend_forecasts'].items():
            report += f"\n{category.replace('_', ' ').title()}:\n"
            for trend in trends:
                report += f"- {trend}\n"
        
        # Accessories Section
        report += "\nACCESSORIES FORECAST\n"
        accessories = forecast['accessories_forecast']
        
        for category, data in accessories['key_accessory_categories'].items():
            report += f"\n{category.upper()}:\n"
            report += f"Trend: {data['trend']}\n"
            report += f"Growth: {data['growth_outlook']}\n"
            report += "Key Items:\n"
            for item in data['key_items']:
                report += f"- {item}\n"
        
        report += "\n---\nGenerated by TrendWeave with Accessories Forecasting"
        return report
    
    def show_training_interface(self):
        """Show model training interface"""
        st.markdown("## 🎯 Model Fine-tuning")
        
        # Training stats
        training_stats = self.data_manager.get_training_stats()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", training_stats['total_samples'])
        with col2:
            st.metric("Sentiment Samples", training_stats['sentiment_samples'])
        with col3:
            st.metric("Source Files", training_stats['files_count'])
        
        if training_stats['total_samples'] < 15:  # 3 files × 5 samples each ≈ 15
            st.warning(f"""
            **📊 More Data Needed for Effective Training**
            
            For optimal fine-tuning, we recommend:
            - At least 3 analyzed PDF documents
            - 15+ training samples
            - Diverse fashion content
            
            Currently available: 
            - {training_stats['total_samples']} training samples from {training_stats['files_count']} documents
            """)
            return
        
        st.info("Fine-tune models on your fashion data for better accuracy")
        
        # Training parameters
        epochs = st.slider("Training Epochs", 1, 5, 3)
        
        if st.button("🚀 Start Fine-tuning", type="primary"):
            with st.spinner("Fine-tuning models..."):
                try:
                    training_data = self.data_manager.load_training_data()
                    
                    # Fine-tune models
                    sentiment_path = self.model_trainer.fine_tune_sentiment_model(training_data, epochs)
                    ner_path = self.model_trainer.fine_tune_ner_model(training_data)
                    embedding_path = self.model_trainer.create_fashion_embedding_model(training_data)
                    
                    # Save metadata
                    training_info = {'samples': len(training_data), 'epochs': epochs}
                    model_paths = {'sentiment': sentiment_path, 'ner': ner_path, 'embeddings': embedding_path}
                    self.model_trainer.save_training_metadata(training_info, model_paths)
                    
                    st.success("✅ Models fine-tuned successfully!")
                    
                    # Reload processor
                    from fashion_processor import FashionProcessor
                    self.processor = FashionProcessor(use_fine_tuned_models=True)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    def show_accessories_dashboard(self):
        """Show dedicated accessories dashboard"""
        st.markdown('<h1 class="main-header">Accessories Dashboard</h1>', unsafe_allow_html=True)
        
        # Sample data for demonstration
        if st.button("Generate Accessories Report"):
            sample_analysis = [
                {
                    'sentiment_analysis': {
                        'overall_sentiment': 'POSITIVE',
                        'confidence': 0.85
                    },
                    'keyword_extraction': {
                        'keywords': [
                            ['sustainable jewelry', 0.9],
                            ['tech accessories', 0.8],
                            ['statement handbags', 0.7],
                            ['designer sunglasses', 0.6],
                            ['smart watches', 0.75],
                            ['vegan leather bags', 0.8]
                        ]
                    },
                    'entity_recognition': {
                        'entities': ['luxury brands', 'sustainable fashion', 'accessory trends']
                    }
                }
            ]
            
            with st.spinner("Generating accessories forecast..."):
                accessory_report = self.forecaster.generate_accessory_specific_report(sample_analysis)
                self.display_accessory_dashboard(accessory_report)
    
    def display_accessory_dashboard(self, accessory_report):
        """Display comprehensive accessories dashboard"""
        
        # Executive Summary
        st.markdown('<h2 class="section-header">📊 Accessories Market Overview</h2>', unsafe_allow_html=True)
        
        summary = accessory_report['executive_summary']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Market Overview", summary['market_overview'])
        with col2:
            st.metric("Overall Outlook", summary['overall_outlook'])
        with col3:
            st.metric("Forecast Period", summary['forecast_period'])
        with col4:
            st.metric("Trending Categories", len(summary['top_trending_categories']))
        
        # Top Trending Categories
        st.subheader("Top Trending Categories")
        for category in summary['top_trending_categories']:
            st.markdown(f'<div class="accessory-card">{category}</div>', unsafe_allow_html=True)
        
        # Growth Drivers
        st.subheader("Growth Drivers")
        for driver in summary['growth_drivers']:
            st.write(f"• {driver}")
        
        # Category Forecasts
        st.markdown('<h2 class="section-header">📈 Category Forecasts</h2>', unsafe_allow_html=True)
        forecasts = accessory_report['category_forecasts']
        
        # Display key categories
        key_categories = forecasts['key_accessory_categories']
        for category, data in key_categories.items():
            with st.expander(f"📦 {category.title()} - {data['growth_outlook']} Growth", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Trend:** {data['trend']}")
                    st.write("**Key Items:**")
                    for item in data['key_items']:
                        st.write(f"- {item}")
                with col2:
                    st.write(f"**Consumer Demand:** {data['consumer_demand']}")
                    st.write(f"**Investment Potential:** High")
        
        # Buying Guide
        st.markdown('<h2 class="section-header">Retail Buying Guide</h2>', unsafe_allow_html=True)
        buying_guide = accessory_report['buying_guide']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Must-Have Categories")
            for category in buying_guide['must_have_categories']:
                st.write(f"{category}")
        
        with col2:
            st.write("Pricing:")
            for strategy in buying_guide['pricing_strategy']:
                st.write(f"{strategy}")
        
        # Risk Assessment
        st.markdown('<h2 class="section-header">Risk Analysis</h2>', unsafe_allow_html=True)
        risk_assessment = accessory_report['risk_assessment']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Market Risks")
            for risk in risk_assessment['market_risks']:
                st.write(f"{risk}")
        
        with col2:
            st.subheader("Mitigation Strategies")
            for strategy in risk_assessment['mitigation_strategies']:
                st.write(f"{strategy}")
        
        with col3:
            st.subheader("Opportunities")
            for opportunity in risk_assessment['opportunities']:
                st.write(f"{opportunity}")

# Run the app
if __name__ == "__main__":
    app = FashionTrendApp()
    app.run()
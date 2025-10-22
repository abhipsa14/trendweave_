# MODEL FINE-TUNING GUIDE

## 🎯 Problem Solved!

Your model wasn't fine-tuning because **NO SENTIMENT DATA** was being collected from PDFs. 

### Root Cause:
1. Sentiment analysis was returning lowercase labels (`'positive'`) 
2. Code expected uppercase labels (`'POSITIVE'`)
3. This caused silent failures - sentiment data was never saved
4. Fine-tuning requires sentiment data but found 0 samples → training failed

### Solution Applied:
✅ Fixed sentiment label case mismatch in `fashion_processor.py`
✅ Created `quick_sentiment_generator.py` to generate sentiment training data
✅ Successfully generated 100 sentiment samples from your PDFs

---

## 📊 Current Status

**Training Data Available:**
- ✅ 31,398 entity samples (for NER)
- ✅ 100 sentiment samples (for sentiment model)
- ✅ 196 PDF files processed
- ✅ Ready for fine-tuning!

---

## 🚀 How to Fine-Tune Your Model

### Option 1: Generate More Sentiment Data (Recommended)

For better fine-tuning results, generate more sentiment samples:

```powershell
# Generate from first 20 PDFs (recommended)
C:/Users/abhip/Desktop/trendweave_/.venv/Scripts/python.exe quick_sentiment_generator.py

# Or edit the script to process more PDFs:
# Change: pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')][:10]
# To:     pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')][:30]
```

**Recommendation:** Aim for 200-500 sentiment samples for optimal fine-tuning

### Option 2: Fine-Tune with Current Data

You can fine-tune now with the 100 samples:

1. **Start your Streamlit app:**
   ```powershell
   streamlit run app.py
   ```

2. **In the app, go to the "Model Training" section**

3. **Click "🚀 Start Fine-tuning"**

4. **Monitor the training progress**

5. **Wait for completion** (may take 5-15 minutes depending on your hardware)

---

## 🔍 Verify Training Data Anytime

Run the diagnostic script:

```powershell
C:/Users/abhip/Desktop/trendweave_/.venv/Scripts/python.exe diagnose_training.py
```

This shows:
- Total sentiment & entity samples
- Data distribution
- Recommendations for improvement

---

## 📈 Understanding the Fine-Tuning Process

### What Gets Fine-Tuned:

1. **Sentiment Model** (RoBERTa)
   - Learns fashion-specific sentiment patterns
   - Uses: 100 sentiment samples from your PDFs
   - Epochs: 3 (configurable in app)

2. **Entity Patterns** (Flair NER)
   - Extracts fashion entities (brands, materials, styles)
   - Uses: 31,398 entity samples

3. **Embedding Model** (SentenceTransformer)
   - Creates fashion-specific embeddings
   - Base model: all-MiniLM-L6-v2

### Training Time:
- **CPU**: 10-20 minutes
- **GPU**: 3-5 minutes (if CUDA available)

---

## 🎓 Best Practices

### For Optimal Results:

1. **More Data = Better Results**
   - Minimum: 100 sentiment samples ✅ (you have this)
   - Recommended: 200-500 samples
   - Ideal: 1000+ samples

2. **Diverse Content**
   - Process PDFs from different seasons
   - Include various fashion categories
   - Mix trend reports, collection reviews, analytics

3. **Regular Updates**
   - Re-generate sentiment data when adding new PDFs
   - Fine-tune periodically (monthly/quarterly)
   - Keep training data current with trends

### Generate More Data:

```python
# Edit quick_sentiment_generator.py to process more PDFs:

# Line 31, change from:
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')][:10]

# To process more (e.g., 50 PDFs):
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')][:50]
```

---

## 🐛 Troubleshooting

### If fine-tuning still fails:

1. **Check sentiment data exists:**
   ```powershell
   C:/Users/abhip/Desktop/trendweave_/.venv/Scripts/python.exe diagnose_training.py
   ```

2. **Verify sentiment samples >= 10:**
   - Need at least 10 samples for training
   - Currently have: 100 ✅

3. **Check for errors in app console:**
   - Look for Python tracebacks
   - Check CUDA/memory issues

4. **Try with fewer epochs:**
   - In app, set epochs = 1 or 2
   - Reduces training time and memory usage

### Common Issues:

- **"No sentiment training data available"** → Run `quick_sentiment_generator.py`
- **Out of memory** → Reduce batch size in `model_trainer.py` (line 98: `per_device_train_batch_size`)
- **Training too slow** → Reduce epochs or use smaller PDF sample

---

## ✅ Success Indicators

After fine-tuning completes, you'll see:

1. **In the app:**
   - ✅ 3/4 or 4/4 Fine-tuned models
   - Model status badges turn green
   - "Models fine-tuned successfully!" message

2. **On disk:**
   - `./fine_tuned_models/fashion_sentiment_model/` (exists)
   - `./fine_tuned_models/fashion_entity_patterns.json` (exists)
   - `./fine_tuned_models/training_metadata.json` (exists)

3. **In analysis:**
   - Sentiment analysis shows "Fine-tuned Fashion Model"
   - Better accuracy on fashion-specific content

---

## 📚 Files Created

- `fashion_processor.py` - Fixed sentiment label handling
- `quick_sentiment_generator.py` - Generate sentiment training data
- `diagnose_training.py` - Diagnose training data issues
- `generate_sentiment_data.py` - Advanced sentiment generator
- `training_data/fashion_training_data_sentiment_*.json` - Your sentiment data

---

## 🎯 Next Steps

1. ✅ Sentiment data generated (100 samples)
2. 🔄 **[OPTIONAL]** Generate more data for better results
3. 🚀 Fine-tune in the Streamlit app
4. 📊 Analyze your fashion PDFs with fine-tuned models
5. 🔄 Periodically retrain as you add more PDFs

---

## 💡 Quick Commands Reference

```powershell
# Generate sentiment data
C:/Users/abhip/Desktop/trendweave_/.venv/Scripts/python.exe quick_sentiment_generator.py

# Check training data status
C:/Users/abhip/Desktop/trendweave_/.venv/Scripts/python.exe diagnose_training.py

# Run the app
streamlit run app.py

# Process specific PDFs (edit script first)
C:/Users/abhip/Desktop/trendweave_/.venv/Scripts/python.exe generate_sentiment_data.py --max-pdfs 20
```

---

**You're all set! 🎉 Your model is now ready to be fine-tuned on your fashion PDF data!**

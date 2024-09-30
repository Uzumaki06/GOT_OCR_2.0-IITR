# GOT_OCR_2.0-IITR

$ GOT (General OCR Theory) 2.0 Model is one of the best models available for the OCR related tasks.In order to implement the GOT_OCR_2.0 Model we need to implement the following steps--:

# Setting the Environment

python -m venv .env

# Installing the libraries for implementing the model

pip install -r requirements.txt

# Now to run the app locally on the system we can use the following command--:

streamlit run app.py

# Conclusion:

I tried fine-tuning it on the hindi text image dataset but the accuracy of the model was not improving and i was hardly getting results. In order to get the desired results we can use more diverse dataset but I was not able to run it on my machine due to hardware limitations.

The GOT-OCR Model used here is the CPU version of the main model and is giving similar results but the time taken is more as compared to the one that uses GPU device. The model performs well and the handles well the OCR Part.

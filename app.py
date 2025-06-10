import os
from flask import Flask, render_template, request, redirect, url_for, session
from instatruth import evaluate_claim
from vidtotext import vid_to_text
from KEYS import ACCESS_KEY, FLASK_SECRET

app = Flask(__name__)
app.secret_key = FLASK_SECRET()

@app.route('/')
def index():
    # Get and clear any error message from session
    error_message = session.pop('error_message', None)
    return render_template('index.html', error_message=error_message)

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api')
def api():
    return render_template('api.html')

@app.route('/results')
def results():
    # Get results from session
    analysis_results = session.get('analysis_results')
    
    if not analysis_results:
        # If no results in session, redirect to home with error message
        return redirect(url_for('index'))
    
    # Render results template with data from session
    return render_template('results.html', **analysis_results)

@app.route('/clear-results')
def clear_results():
    # Clear analysis results from session
    session.pop('analysis_results', None)
    return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        video_url = request.form.get('video_url')
        access_key = request.form.get('access_key')
        
        if not video_url:
            session['error_message'] = 'No video URL provided'
            return redirect(url_for('index'))
        
        if not access_key:
            session['error_message'] = 'Access key is required'
            return redirect(url_for('index'))
        
        # Validate access key
        if access_key != ACCESS_KEY():
            session['error_message'] = 'Invalid access key'
            return redirect(url_for('index'))

        # Step 1: Transcribe video to text
        transcription = vid_to_text(video_url)
        if not transcription:
            session['error_message'] = 'Failed to transcribe video'
            return redirect(url_for('index'))
        
        # Step 2: Run full fact-checking analysis
        analysis_result = evaluate_claim(transcription)
        if not analysis_result:
            session['error_message'] = 'Failed to analyze transcription'
            return redirect(url_for('index'))

        # Step 3: Store results in session and redirect
        truth_score_percent = int(analysis_result.get('combined_score', 0) * 100)
        
        # Store results in session for persistence across reloads
        session['analysis_results'] = {
            'transcription': transcription,
            'truth_score': truth_score_percent,
            'final_label': analysis_result.get('final_label', 'inconclusive'),
            'bert_score': int(analysis_result.get('bert_score', 0) * 100),
            'factcheck_score': int(analysis_result.get('factcheck_score', 0) * 100),
            'summary': analysis_result.get('summary', 'No summary available'),
            'claims_analyzed': analysis_result.get('claims_analyzed', 0),
            'individual_claims': analysis_result.get('individual_claims', [])
        }
        
        return redirect(url_for('results'))
        
    except Exception as e:
        # Store error in session and redirect to index with error
        session['error_message'] = f'Analysis failed: {str(e)}'
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False)
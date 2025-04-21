from flask import Flask, render_template, request, jsonify
import instatruth

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api')
def api():
    return render_template('api.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        video_url = request.form.get('video_url')

        # check if tt or instagram
        if 'tiktok.com' in video_url:
            transcription = instatruth.tt_to_text(video_url)
        elif 'instagram.com' in video_url:
            transcription = instatruth.reel_to_text(video_url)
        else:
            return jsonify({'error': 'Bad URL'}), 400
        
        if not transcription:
            return jsonify({'error': 'Failed to transcribe video'}), 400
        
        # do more analysis here
        
        # jsonify result
        return jsonify({
            'success': True,
            'transcription': transcription,
            'truth_score': 69,  # Placeholder score
            'sources': [
                {
                    'name': 'Example Source',
                    'verified': True,
                    'description': 'This is a placeholder for verification sources'
                }
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
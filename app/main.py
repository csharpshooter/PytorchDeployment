from flask import Flask, request, jsonify, render_template, redirect
import traceback,sys,os

from app.torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            class_id,class_name = get_prediction(tensor)
            return render_template('result.html', class_id=class_id,
                class_name=class_name)            
            #return jsonify(data)
        except :
            return jsonify({'error': traceback.print_exception(*sys.exc_info())})
    return render_template('index.html')
    
    
if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/compute', methods=['POST'])
def compute():
    data = request.json
    x = data['x']
    return jsonify(result=x**2)

if __name__ == '__main__':
    app.run(port=5000)
    
    
# In MATLAB use
# url = 'http://127.0.0.1:5000/compute';
# data = jsonencode(struct('x', 4));
# options = weboptions('MediaType', 'application/json');
# response = webwrite(url, data, options);
# result = response.result;
# disp(result);  % Displays: 16
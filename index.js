var spawn = require('child_process').spawn;
var data = [
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ],
    [
        [0],
        [0],
        [1],
        [1]
    ]
];

/**
 * Train /tmp simple model
 * @param {dData} matrices 
 */
var train = (dData, callback) => {
    var py = spawn('python', ['gradient_descent_nn.py']);
    var dataString = '';

    console.log('Training...');

    py.stdout.on('data', (_data) => {
        dataString += _data.toString();
    });

    py.stdout.on('end', () => {
        console.log(dataString);

        if (callback) {
            callback();
        }
    });

    py.stdin.write(JSON.stringify(dData));

    py.stdin.end();
}

/**
 * Predict tmp/ simple model
 */
var predict = () => {
    var prediction = '';
    var predictionParsed;
    var pyModel = spawn('python', ['load_model.py']);

    console.log('Done.');
    console.log('Making prediction...');

    pyModel.stdout.on('data', (_data) => {
        prediction += _data.toString();
    })

    pyModel.stdout.on('end', () => {
        console.log('Prediction: ', prediction);
    });

    pyModel.stdin.write(JSON.stringify(data));

    pyModel.stdin.end();
}

train(data, predict);
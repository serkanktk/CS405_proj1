function multiplyMatrices(matrixA, matrixB) {
    var result = [];

    for (var i = 0; i < 4; i++) {
        result[i] = [];
        for (var j = 0; j < 4; j++) {
            var sum = 0;
            for (var k = 0; k < 4; k++) {
                sum += matrixA[i * 4 + k] * matrixB[k * 4 + j];
            }
            result[i][j] = sum;
        }
    }

    // Flatten the result array
    return result.reduce((a, b) => a.concat(b), []);
}
function createIdentityMatrix() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
}
function createScaleMatrix(scale_x, scale_y, scale_z) {
    return new Float32Array([
        scale_x, 0, 0, 0,
        0, scale_y, 0, 0,
        0, 0, scale_z, 0,
        0, 0, 0, 1
    ]);
}

function createTranslationMatrix(x_amount, y_amount, z_amount) {
    return new Float32Array([
        1, 0, 0, x_amount,
        0, 1, 0, y_amount,
        0, 0, 1, z_amount,
        0, 0, 0, 1
    ]);
}

function createRotationMatrix_Z(radian) {
    return new Float32Array([
        Math.cos(radian), -Math.sin(radian), 0, 0,
        Math.sin(radian), Math.cos(radian), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_X(radian) {
    return new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(radian), -Math.sin(radian), 0,
        0, Math.sin(radian), Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_Y(radian) {
    return new Float32Array([
        Math.cos(radian), 0, Math.sin(radian), 0,
        0, 1, 0, 0,
        -Math.sin(radian), 0, Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function getTransposeMatrix(matrix) {
    return new Float32Array([
        matrix[0], matrix[4], matrix[8], matrix[12],
        matrix[1], matrix[5], matrix[9], matrix[13],
        matrix[2], matrix[6], matrix[10], matrix[14],
        matrix[3], matrix[7], matrix[11], matrix[15]
    ]);
}

const vertexShaderSource = `
attribute vec3 position;
attribute vec3 normal; // Normal vector for lighting

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;

uniform vec3 lightDirection;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vNormal = vec3(normalMatrix * vec4(normal, 0.0));
    vLightDirection = lightDirection;

    gl_Position = vec4(position, 1.0) * projectionMatrix * modelViewMatrix; 
}

`

const fragmentShaderSource = `
precision mediump float;

uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(vLightDirection);
    
    // Ambient component
    vec3 ambient = ambientColor;

    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // Specular component (view-dependent)
    vec3 viewDir = vec3(0.0, 0.0, 1.0); // Assuming the view direction is along the z-axis
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}

`

/**
 * @WARNING DO NOT CHANGE ANYTHING ABOVE THIS LINE
 */



/**
 * 
 * @TASK1 Calculate the model view matrix by using the chatGPT
 */

function getChatGPTModelViewMatrix() {
    const transformationMatrix = new Float32Array([
    0.3298698, -0.21779788, 0.61237246, 0.3,
    0.375, 0.21650635, -0.5, -0.25,
    -0.02368359, 0.39457455, 0.61237246, 0,
    0, 0, 0, 1

    ]);
    return getTransposeMatrix(transformationMatrix);
}


/**
 * 
 * @TASK2 Calculate the model view matrix by using the given 
 * transformation methods and required transformation parameters
 * stated in transformation-prompt.txt
 */
function getModelViewMatrix() {
	const Calculationofang = (angle) => angle * Math.PI / 180;

    const ex = new Float32Array(16);// Temporary array for intermediate results
    const fla = new Float32Array(16);
    
    const rotationXMatrix = new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(Calculationofang(30)), -Math.sin(Calculationofang(30)), 0,
        0, Math.sin(Calculationofang(30)), Math.cos(Calculationofang(30)), 0,
        0, 0, 0, 1
    ]);
    const translationMatrix = new Float32Array([
        1, 0, 0, 0.3,
        0, 1, 0, -0.25,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
    
    const scalingMatrix = new Float32Array([
        0.5, 0, 0, 0,
        0, 0.5, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
    const rotationYMatrix = new Float32Array([
        Math.cos(Calculationofang(45)), 0, Math.sin(Calculationofang(45)), 0,
        0, 1, 0, 0,
        -Math.sin(Calculationofang(45)), 0, Math.cos(Calculationofang(45)), 0,
        0, 0, 0, 1
    ]);
    
    const rotationZMatrix = new Float32Array([
        Math.cos(Calculationofang(60)), -Math.sin(Calculationofang(60)), 0, 0,
        Math.sin(Calculationofang(60)), Math.cos(Calculationofang(60)), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
    

    
    const TransformationwithMultp = (nv1, mt1, mt2) => {
        for (let x = 0; x < 4; x++) {
			
            for (let j = 0; j < 4; j++) {
				
                ex[j * 4 + x] = mt1[j * 4] * mt2[x] +mt1[j * 4 + 1] * mt2[x + 4] + mt1[j * 4 + 2] * mt2[x + 8] + mt1[j * 4 + 3] * mt2[x + 12];
            }
        }
        for (let x = 0; x < 16; x++) {
            nv1[x] = ex[x]; // Copying the results from the temporary array
        }
    };
    
    TransformationwithMultp(ex, translationMatrix, scalingMatrix);
    TransformationwithMultp(fla, ex, rotationXMatrix);
    TransformationwithMultp(ex, fla, rotationYMatrix);
    TransformationwithMultp(fla, ex, rotationZMatrix);
    
    const CalculatedtransformationMatrix = new Float32Array(fla);
	
	
	
    return CalculatedtransformationMatrix;
    // calculate the model view matrix by using the transformation
    // methods and return the modelView matrix in this method
}

/**
 * 
 * @TASK3 Ask CHAT-GPT to animate the transformation calculated in 
 * task2 infinitely with a period of 10 seconds. 
 * First 5 seconds, the cube should transform from its initial 
 * position to the target position.
 * The next 5 seconds, the cube should return to its initial position.
 */
function getPeriodicMovement(startTime) {
    const currentTime = Date.now();
    const elapsed = (currentTime - startTime) / 1000; // Time in seconds
    const period = 10; // Total period of the animation cycle
    const halfPeriod = period / 2;
    const t = (elapsed % period) / period; // Normalized time in the range [0, 1]

    let interpolationFactor;
    if (t < 0.5) {
        // First half: Interpolate from initial to transformed state
        interpolationFactor = t * 2; // Normalized time [0, 1] for the first half
    } else {
        // Second half: Interpolate from transformed state back to initial
        interpolationFactor = (1 - t) * 2; // Normalized time [0, 1] for the second half
    }

    const initialMatrix = getChatGPTModelViewMatrix(); // Assuming this is the initial state matrix
    const finalMatrix = getModelViewMatrix(); // The transformation matrix from Task 2

    const interpolatedMatrix = new Float32Array(16);
    for (let i = 0; i < 16; i++) {
        // Linearly interpolate each element of the matrix
        interpolatedMatrix[i] = initialMatrix[i] * (1 - interpolationFactor) + finalMatrix[i] * interpolationFactor;
    }

    return interpolatedMatrix;
}





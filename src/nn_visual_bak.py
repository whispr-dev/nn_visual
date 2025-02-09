import React, {useState, useEffect} from 'react';

const NeuralNetworkVis = () => {
  const [time, setTime] = useState(0);
  const [values, setValues] = useState({
    inputs: [20, 10, -30, 40],
    layerOne: [0, 0],
    output: [0]
  });

  // Analog activation function (-100 to +100)
  const analogActivation = (x, theta = 0) => {
    const shifted = x - theta;
    return Math.max(-100, Math.min(100, shifted));
  };

  // Weight matrices from the document
  const weights = {
    layer1: [
      [0.5, 1.0, 0, 0],    // Neuron A weights
      [0, 0, -1.5, 0.75]   // Neuron B weights
    ],
    layer2: [
      [1.0, -0.5]          // Output neuron weights
    ]
  };

  const thresholds = {
    layer1: [15, 50],
    layer2: [-5]
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(prev => (prev + 1) % 100);
      
      // Generate oscillating inputs
      const newInputs = values.inputs.map((val, i) => 
        20 * Math.sin(time / 10 + i) + val
      );

      // Forward propagation
      const layer1 = weights.layer1.map((neuronWeights, i) => {
        const sum = neuronWeights.reduce((acc, w, j) => 
          acc + w * newInputs[j], 0
        );
        return analogActivation(sum, thresholds.layer1[i]);
      });

      const output = weights.layer2[0].reduce((acc, w, i) => 
        acc + w * layer1[i], 0
      );
      const finalOutput = [analogActivation(output, thresholds.layer2[0])];

      setValues({
        inputs: newInputs,
        layerOne: layer1,
        output: finalOutput
      });
    }, 50);

    return () => clearInterval(timer);
  }, [time]);

  const NeuronVis = ({ value, x, y }) => {
    const normalizedValue = (value + 100) / 200; // Convert -100 to +100 to 0 to 1
    const intensity = Math.floor(normalizedValue * 255);
    
    return (
      <div 
        className="absolute rounded-full w-8 h-8 flex items-center justify-center text-xs"
        style={{
          left: `${x}%`,
          top: `${y}%`,
          backgroundColor: `rgb(${255-intensity}, ${255-intensity}, 255)`,
          color: intensity > 128 ? 'white' : 'black',
          transform: 'translate(-50%, -50%)'
        }}
      >
        {Math.round(value)}
      </div>
    );
  };

  const ConnectionLine = ({ startX, startY, endX, endY, weight }) => {
    const strokeWidth = Math.abs(weight) * 2;
    const opacity = Math.abs(weight);
    
    return (
      <line
        x1={`${startX}%`}
        y1={`${startY}%`}
        x2={`${endX}%`}
        y2={`${endY}%`}
        stroke={weight >= 0 ? "#4444ff" : "#ff4444"}
        strokeWidth={strokeWidth}
        opacity={opacity}
      />
    );
  };

  return (
    <div className="w-full h-96 bg-gray-100 relative">
      <svg className="absolute w-full h-full">
        {/* Input to Layer 1 connections */}
        {values.inputs.map((_, inputIdx) => 
          weights.layer1.map((neuronWeights, neuronIdx) => (
            <ConnectionLine
              key={`input-${inputIdx}-l1-${neuronIdx}`}
              startX={15 + inputIdx * 20}
              startY={20 + inputIdx * 15}
              endX={50}
              endY={30 + neuronIdx * 30}
              weight={neuronWeights[inputIdx]}
            />
          ))
        )}
        
        {/* Layer 1 to Output connections */}
        {values.layerOne.map((_, idx) => (
          <ConnectionLine
            key={`l1-${idx}-output`}
            startX={50}
            startY={30 + idx * 30}
            endX={85}
            endY={50}
            weight={weights.layer2[0][idx]}
          />
        ))}
      </svg>

      {/* Input neurons */}
      {values.inputs.map((val, idx) => (
        <NeuronVis
          key={`input-${idx}`}
          value={val}
          x={15 + idx * 20}
          y={20 + idx * 15}
        />
      ))}

      {/* Layer 1 neurons */}
      {values.layerOne.map((val, idx) => (
        <NeuronVis
          key={`l1-${idx}`}
          value={val}
          x={50}
          y={30 + idx * 30}
        />
      ))}

      {/* Output neuron */}
      <NeuronVis
        value={values.output[0]}
        x={85}
        y={50}
      />
    </div>
  );
};

export default NeuralNetworkVis;
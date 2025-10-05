Place TensorFlow.js WASM binaries in this folder for GitHub Pages deployment.

Files expected (from @tensorflow/tfjs-backend-wasm package):
- tfjs-backend-wasm.wasm
- tfjs-backend-wasm.wasm.simd
- tfjs-backend-wasm.wasm.threaded.simd

We configure the runtime to load from /Simulation/tfwasm/ via wasm.setWasmPaths().
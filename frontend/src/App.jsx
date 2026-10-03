import { useEffect, useState } from "react"; //pulls two tools from react library

function App() {
  //sets the state of the variables for current and a function to change it 
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/health") //sends a GET request to /health
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Server responded with status ${response.status}`); //throw error if the status code isn't 200
        }
        return response.json(); //if the status code is 200, it turns the json text into a JS object
      })
      .then((data) => setHealth(data)) //data is now {status: "ok"}
      .catch((err) => setError(err.message)) //if there was an error
      .finally(() => setLoading(false)); //sets loading to false since we're done
  }, []);

  //output that appears on the screen
  return (
    <main style={{ fontFamily: "sans-serif", padding: "2rem" }}> 
      <h1>Backend health check</h1>
      {loading && <p>Checking the backend…</p>}

      {error && (
        <p style={{ color: "crimson" }}>
          Couldn't reach the backend: {error}. Is uvicorn running on port 8000?
        </p>
      )}

      {health && (
        <pre>{JSON.stringify(health, null, 2)}</pre>
      )}
    </main>
  );
}

export default App; //makes app available to other files

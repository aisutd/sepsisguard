import PatientList from "./components/PatientList"
import RiskScore from "./components/RiskScore"
import ShapChart from "./components/ShapChart"
import SummaryCard from "./components/SummaryCard"

import "./App.css"

function App() {
  return(
    <div className="dashboard">

      <header className="dashboard-header">
        <h1>SepsisGuard</h1>
      </header>

      <aside className="sidebar">
        <PatientList/>
      </aside>

      <main className="detail">
        <div className="detail-row">
          <RiskScore />
          <SummaryCard />
        </div>

        <ShapChart />
      </main>

    </div>
  );
}

export default App; 

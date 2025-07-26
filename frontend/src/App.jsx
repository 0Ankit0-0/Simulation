import React from "react";
import CaseReview from "./component/caseReview";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import CaseSubmit from "./component/caseSubmit";

function App() {
  return (
    <div>
      <h1 className="text-3xl bg-amber-300 p-4 text-center">
        Welcome to the Courtroom Simulation
      </h1>
      <Router>
        <Routes>
          <Route path="/" element={<CaseSubmit />} />
          <Route path="/review/:caseId" element={<CaseReview />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
// import logo from './logo.svg';
// import './App.css';
import './ai_styles.css';

function App() {
  return (
    <div className="app-container">
      <div className="container-fluid">
        <div className="row">
          <div className="col-3 bg-light sidebar">
            <input type="text" className="form-control mb-3" placeholder="Search" />
            <div className="d-flex align-items-center mb-3">
              {/* Using placeholder.com for temporary avatar */}
              <img src="https://via.placeholder.com/40" alt="User" className="rounded-circle" width="40" />
              <div className="ms-2">
                <strong>Nancy Fernandez</strong>
                <p>Hi Jordan! Feels like it&apos;s ...</p>
              </div>
            </div>
          </div>

          <div className="col-6 p-3">
            <div className="d-flex align-items-start mb-3">
              <img src="https://via.placeholder.com/40" alt="User" className="rounded-circle" width="40" />
              <div className="ms-2 message bg-light">
                Hi Jordan! Feels like it&apos;s been a while.
              </div>
            </div>
            <div className="d-flex align-items-start mb-3 justify-content-end">
              <div className="me-2 message bg-primary text-white">
                Yes, it has. How have you been?
              </div>
              <img src="https://via.placeholder.com/40" alt="User" className="rounded-circle" width="40" />
            </div>
          </div>

          <div className="col-3 bg-light p-3">
            <h5>Your files</h5>
            <div className="mb-3">
              <p><a href="#" className="text-primary">All Files</a></p>
              <p><a href="#" className="text-primary">Notification preferences</a></p>
              <div className="mb-3">
                <img src="https://via.placeholder.com/40" alt="User" className="rounded-circle" width="40" />
                <a href="#" className="ms-2 text-dark">View Nancy&apos;s profile</a>
              </div>
              <p><a href="#" className="text-danger">Delete conversation</a></p>
            </div>
            <div className="profile-options">
              <h5>Nancy Fernandez</h5>
              <p>Online</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}



export default App;

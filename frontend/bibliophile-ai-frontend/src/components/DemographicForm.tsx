import { useState } from "react";

type DemographicProps = {
  initialPincode?: string;
  initialAge?: number;
  onSave: (values: { pincode: string; age: number }) => void;
  onBack: () => void;
  loading?: boolean;
};

export default function DemographicForm({
  initialPincode = "",
  initialAge,
  onSave,
  onBack,
  loading = false,
}: DemographicProps) {
  const [pincode, setPincode] = useState(initialPincode);
  const [age, setAge] = useState<number | "">(initialAge ?? "");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!pincode || !age) return;
    onSave({ pincode, age: Number(age) });
  };

  const isValid = pincode.length >= 5 && age !== "" && Number(age) >= 10 && Number(age) <= 120;

  return (
    <div
      className="min-vh-100 d-flex align-items-center justify-content-center position-relative"
      style={{
        background: 'linear-gradient(135deg, #f093fb 0%, #764ba2 50%, #667eea 100%)',
        padding: '2rem 0'
      }}
    >
      {/* Animated background pattern */}
      <div
        className="position-absolute w-100 h-100 opacity-10"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.4'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3Ccircle cx='15' cy='15' r='2'/%3E%3Ccircle cx='45' cy='45' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          animation: 'float 20s ease-in-out infinite',
          pointerEvents: 'none'
        }}
      />

      {/* Floating decorative elements */}
      <div
        className="position-absolute"
        style={{ top: '12%', left: '6%', animation: 'bounce 4s infinite', pointerEvents: 'none' }}
      >
        <div className="rounded-circle bg-white bg-opacity-15 p-3" style={{ backdropFilter: 'blur(10px)' }}>
          <span style={{ fontSize: '2rem' }}>🌍</span>
        </div>
      </div>
      <div
        className="position-absolute"
        style={{ top: '22%', right: '10%', animation: 'bounce 4s infinite 1s', pointerEvents: 'none' }}
      >
        <div className="rounded-circle bg-white bg-opacity-15 p-2" style={{ backdropFilter: 'blur(10px)' }}>
          <span style={{ fontSize: '1.5rem' }}>✨</span>
        </div>
      </div>
      <div
        className="position-absolute"
        style={{ bottom: '18%', left: '10%', animation: 'bounce 4s infinite 2s', pointerEvents: 'none' }}
      >
        <div className="rounded-circle bg-white bg-opacity-15 p-2" style={{ backdropFilter: 'blur(10px)' }}>
          <span style={{ fontSize: '1.5rem' }}>🎯</span>
        </div>
      </div>

      <div className="container">
        <div className="row justify-content-center">
          <div className="col-lg-6 col-md-8">
            <div
              className="card shadow-lg border-0 position-relative overflow-hidden"
              style={{
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(20px)',
                borderRadius: '2rem',
                border: '1px solid rgba(255, 255, 255, 0.2)'
              }}
            >
              {/* Card Header */}
              <div
                className="card-header border-0 text-center py-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
                  borderRadius: '2rem 2rem 0 0'
                }}
              >
                <div className="mb-3">
                  <span style={{ fontSize: '3rem', filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))' }}>
                    🎯✨
                  </span>
                </div>
                <h2
                  className="display-6 fw-bold mb-2"
                  style={{
                    background: 'linear-gradient(135deg, #f093fb, #764ba2)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text'
                  }}
                >
                  Almost There!
                </h2>
                <p className="lead text-muted mb-0">
                  Help us personalise your experience with a few last details
                </p>
              </div>

              {/* Card Body */}
              <form className="card-body p-4" onSubmit={handleSubmit}>
                <div className="mb-4">
                  <label
                    className="form-label fw-bold mb-2"
                    style={{ color: '#764ba2', fontSize: '0.95rem' }}
                  >
                    📍 Pincode
                  </label>
                  <input
                    className="form-control"
                    value={pincode}
                    onChange={e => setPincode(e.target.value)}
                    required
                    minLength={5}
                    maxLength={10}
                    disabled={loading}
                    placeholder="Enter your pincode"
                    style={{
                      borderRadius: '0.75rem',
                      border: '2px solid',
                      borderColor: pincode.length >= 5 ? '#28a745' : '#667eea',
                      padding: '0.75rem 1rem',
                      fontSize: '1rem',
                      transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
                      background: '#f8f9fa',
                      color: '#1a1a1a'
                    }}
                    onFocus={e => { e.target.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.2)' }}
                    onBlur={e => { e.target.style.boxShadow = 'none' }}
                  />
                </div>

                <div className="mb-4">
                  <label
                    className="form-label fw-bold mb-2"
                    style={{ color: '#764ba2', fontSize: '0.95rem' }}
                  >
                    🎂 Age
                  </label>
                  <input
                    className="form-control"
                    type="number"
                    min={10}
                    max={120}
                    value={age}
                    onChange={e => setAge(Number(e.target.value))}
                    required
                    disabled={loading}
                    placeholder="Enter your age"
                    style={{
                      borderRadius: '0.75rem',
                      border: '2px solid',
                      borderColor: age !== "" && Number(age) >= 10 ? '#28a745' : '#667eea',
                      padding: '0.75rem 1rem',
                      fontSize: '1rem',
                      transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
                      background: '#f8f9fa',
                      color: '#1a1a1a'
                    }}
                    onFocus={e => { e.target.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.2)' }}
                    onBlur={e => { e.target.style.boxShadow = 'none' }}
                  />
                </div>

                {/* Action Buttons */}
                <div className="d-flex justify-content-center gap-3 mt-2">
                  <button
                    className="btn btn-outline-secondary btn-lg px-4 py-2 rounded-pill fw-bold"
                    type="button"
                    onClick={onBack}
                    disabled={loading}
                    style={{ minWidth: '120px', transition: 'all 0.3s ease' }}
                  >
                    ← Back
                  </button>
                  <button
                    className={`btn btn-lg px-5 py-2 rounded-pill fw-bold position-relative overflow-hidden ${
                      !isValid ? 'btn-outline-secondary' : 'btn-success'
                    }`}
                    type="submit"
                    disabled={!isValid || loading}
                    style={{
                      minWidth: '160px',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      boxShadow: isValid && !loading ? '0 8px 25px rgba(40, 167, 69, 0.3)' : 'none',
                      background: isValid && !loading
                        ? 'linear-gradient(135deg, #28a745, #20c997)'
                        : undefined,
                      transform: isValid && !loading ? 'translateY(-2px)' : 'none'
                    }}
                  >
                    {loading && (
                      <span
                        className="spinner-border spinner-border-sm me-2"
                        style={{ animation: 'spin 1s linear infinite' }}
                      />
                    )}
                    <span>
                      {loading ? "Saving..." : "🚀 Finish!"}
                    </span>
                    {isValid && !loading && (
                      <div
                        className="position-absolute top-0 start-0 w-100 h-100"
                        style={{
                          background: 'linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.3) 50%, transparent 70%)',
                          borderRadius: 'inherit',
                          animation: 'shine 3s ease-in-out infinite',
                          pointerEvents: 'none'
                        }}
                      />
                    )}
                  </button>
                </div>

                {!isValid && (
                  <p className="text-muted small text-center mt-3 mb-0">
                    💡 Fill in both fields to complete your profile
                  </p>
                )}
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

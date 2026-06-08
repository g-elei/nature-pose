import { useState } from 'react';
import { HomeScreen } from './components/screens/HomeScreen';
import { SessionScreen } from './components/screens/SessionScreen';
import { DashboardScreen } from './components/screens/DashboardScreen';

type Screen = 'home' | 'session' | 'dashboard';

function App() {
  const [screen, setScreen] = useState<Screen>('home');

  return (
    <>
      {screen === 'home' && <HomeScreen onStart={() => setScreen('session')} />}
      {screen === 'session' && <SessionScreen onFinish={() => setScreen('dashboard')} />}
      {screen === 'dashboard' && <DashboardScreen onBack={() => setScreen('home')} />}
    </>
  );
}

export default App;
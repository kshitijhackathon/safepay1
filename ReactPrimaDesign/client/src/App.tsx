import React from "react";
import { Switch, Route } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import { AuthProvider } from "@/hooks/use-auth-state";
import { ThemeProvider } from "@/contexts/ThemeContext";
import NotFound from "@/pages/not-found";
import Login from "@/pages/login";
import OTP from "@/pages/otp";
import PIN from "@/pages/pin";
import PhoneLogin from "@/pages/phone-login";
import SetupSecurity from "@/pages/setup-security";
import Home from "@/pages/home";
import Help from "@/pages/help";
import Scan from "@/pages/scan";
import Payment from "@/pages/payment";
import Checkout from "@/pages/checkout";
import Success from "@/pages/success";
import UpiCheck from "@/pages/upi-check";
import ReportScam from "@/pages/report-scam";
import VoiceCheck from "@/pages/voice-check";
import MessageCheck from "@/pages/message-check";
import WhatsAppCheck from "@/pages/whatsapp-check";
import ChatSupport from "@/pages/chat-support";
import Account from "@/pages/account";
import Profile from "@/pages/profile";
import PaymentMethods from "@/pages/payment-methods";
import Settings from "@/pages/settings";
import HelpSupport from "@/pages/help-support";
import ScamNews from "@/pages/scam-news";
import LegalHelp from "@/pages/legal-help";
import FraudHeatmap from "@/pages/fraud-heatmap";
import FraudHeatmapBasic from "@/pages/fraud-heatmap-basic";
import FraudMap from "@/pages/fraud-map";
import FraudMapPage from "@/pages/fraud-map-page";
import ConfirmTransaction from "@/pages/confirm-transaction";
import HistoryPage from "@/pages/history-new";
import RiskScoreDemo from "@/pages/risk-score-demo";
import SecuritySettings from "@/pages/security-settings";
import MyReports from "@/pages/my-reports";
import VideoDetectionPage from "@/pages/video-detection-page";
import ScamDetectionPage from "@/pages/scam-detection-page";
import QRScan from "@/pages/qr-scan";
import ManualUpiEntry from "@/pages/manual-upi-entry";
import AllServices from "@/pages/all-services";
import VideoCheck from "@/pages/video-check";
import CallCheck from "@/pages/call-check";
import MainLayout from "@/layouts/main-layout";

function Router() {
  return (
    <Switch>
      <Route path="/" component={PhoneLogin} />
      <Route path="/phone-login" component={PhoneLogin} />
      <Route path="/login" component={Login} />
      <Route path="/otp" component={OTP} />
      <Route path="/pin" component={PIN} />
      <Route path="/setup-security" component={SetupSecurity} />
      <Route path="/home">
        {() => (
          <MainLayout>
            <Home />
          </MainLayout>
        )}
      </Route>
      <Route path="/scan">
        {() => <Scan />}
      </Route>
      <Route path="/qr-scan" component={QRScan} />
      <Route path="/manual-upi-entry" component={ManualUpiEntry} />
      <Route path="/simplified-scan" component={Scan} />
      <Route path="/confirm-transaction" component={ConfirmTransaction} />
      <Route path="/payment" component={Payment} />
      <Route path="/checkout" component={Checkout} /> 
      <Route path="/success" component={Success} />
      <Route path="/upi-check">
        {() => (
          <MainLayout>
            <UpiCheck />
          </MainLayout>
        )}
      </Route>
      <Route path="/report-scam">
        {() => (
          <MainLayout>
            <ReportScam />
          </MainLayout>
        )}
      </Route>
      <Route path="/voice-check">
        {() => (
          <MainLayout>
            <VoiceCheck />
          </MainLayout>
        )}
      </Route>
      <Route path="/message-check">
        {() => (
          <MainLayout>
            <MessageCheck />
          </MainLayout>
        )}
      </Route>
      <Route path="/whatsapp-check">
        {() => (
          <MainLayout>
            <WhatsAppCheck />
          </MainLayout>
        )}
      </Route>
      <Route path="/history">
        {() => (
          <MainLayout>
            <HistoryPage />
          </MainLayout>
        )}
      </Route>
      <Route path="/chat-support">
        {() => (
          <MainLayout>
            <ChatSupport />
          </MainLayout>
        )}
      </Route>
      <Route path="/account">
        {() => (
          <MainLayout>
            <Account />
          </MainLayout>
        )}
      </Route>
      <Route path="/scam-news">
        {() => (
          <MainLayout>
            <ScamNews />
          </MainLayout>
        )}
      </Route>
      <Route path="/legal-help">
        {() => (
          <MainLayout>
            <LegalHelp />
          </MainLayout>
        )}
      </Route>
      <Route path="/fraud-heatmap">
        {() => (
          <MainLayout>
            <FraudHeatmap />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/profile">
        {() => (
          <MainLayout>
            <Profile />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/payment-methods">
        {() => (
          <MainLayout>
            <PaymentMethods />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/my-reports">
        {() => (
          <MainLayout>
            <MyReports />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/settings">
        {() => (
          <MainLayout>
            <Settings />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/help-support">
        {() => (
          <MainLayout>
            <HelpSupport />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/help" component={Help} />
      
      <Route path="/fraud-heatmap-basic" component={FraudHeatmapBasic} />
      
      <Route path="/fraud-map">
        {() => (
          <MainLayout>
            <FraudMap />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/fraud-map-3d">
        {() => (
          <MainLayout>
            <FraudMapPage />
          </MainLayout>
        )}
      </Route>

      <Route path="/risk-score-demo">
        {() => (
          <MainLayout>
            <RiskScoreDemo />
          </MainLayout>
        )}
      </Route>

      <Route path="/security-settings">
        {() => (
          <MainLayout>
            <SecuritySettings />
          </MainLayout>
        )}
      </Route>

      <Route path="/video-detection">
        {() => (
          <MainLayout>
            <VideoDetectionPage />
          </MainLayout>
        )}
      </Route>

      <Route path="/scam-detection">
        {() => (
          <MainLayout>
            <ScamDetectionPage />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/all-services">
        {() => (
          <MainLayout>
            <AllServices />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/video-check">
        {() => (
          <MainLayout>
            <VideoCheck />
          </MainLayout>
        )}
      </Route>
      
      <Route path="/call-check">
        {() => (
          <MainLayout>
            <CallCheck />
          </MainLayout>
        )}
      </Route>

      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <Router />
          <Toaster />
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
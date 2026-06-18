import React from 'react';

interface State { hasError: boolean; message: string | null }
interface Props {
  eyebrow?: string;
  title?: string;
  retryLabel?: string;
  children?: React.ReactNode;
}

export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false, message: null };

  static getDerivedStateFromError(error: unknown): State {
    return { hasError: true, message: error instanceof Error ? error.message : 'Unknown rendering error.' };
  }

  componentDidCatch(error: unknown) {
    console.error('WaferVision UI boundary caught an error:', error);
  }

  render() {
    if (!this.state.hasError) return this.props.children;
    return <section className="card error-boundary"><p className="eyebrow">{this.props.eyebrow ?? 'UI safety boundary'}</p><h2>{this.props.title ?? 'Panel failed safely'}</h2><p>{this.state.message}</p><button className="ghost" onClick={() => this.setState({ hasError: false, message: null })}>{this.props.retryLabel ?? 'Retry panel'}</button></section>;
  }
}

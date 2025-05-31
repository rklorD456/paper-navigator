# Paper Navigator Documentation

## Overview
Paper Navigator is a web application designed to help researchers and academics manage their paper reading workflow. It provides personalized paper recommendations, reading status tracking, and a modern interface for organizing academic papers.

## Features

### Paper Management
- Add and organize academic papers
- Track reading status (unread, reading, completed)
- Add personal notes and summaries
- Store key takeaways from papers

### Recommendation System
- AI-powered paper recommendations
- Personalized based on user preferences
- Considers citation count and recency
- Keyword-based matching

### User Experience
- Modern, responsive design
- Dark mode support
- Keyboard shortcuts
- Real-time updates

## API Documentation

### Authentication Endpoints
- `POST /api/register` - Register new user
- `POST /api/login` - User login
- `POST /api/logout` - User logout
- `POST /api/reset-password` - Password reset

### Paper Endpoints
- `GET /papers` - List all papers
- `GET /papers/{id}` - Get paper details
- `POST /api/rate-paper` - Rate a paper
- `POST /api/update-reading` - Update reading status

### User Profile Endpoints
- `GET /api/user-profile` - Get user preferences
- `POST /api/update-preferences` - Update user preferences

## Development Guide

### Prerequisites
- Python 3.8+
- Node.js 14+
- SQLite or PostgreSQL

### Local Development
1. Clone the repository
2. Set up virtual environment
3. Install dependencies
4. Configure environment variables
5. Run development server

### Testing
- Unit tests: `pytest`
- Integration tests: `pytest integration`
- Coverage report: `pytest --cov`

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details. 
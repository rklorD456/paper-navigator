# Paper Navigator

A web application for managing and organizing academic papers, built with FastAPI and modern web technologies.

## Features

- Paper management and organization
- Paper recommendations based on user preferences
- Reading status tracking
- Paper rating system
- Keyword-based paper filtering
- User authentication and session management
- Responsive design with dark mode support

## Tech Stack

- Backend: FastAPI, SQLAlchemy
- Frontend: HTML, TailwindCSS, JavaScript
- Database: SQLite (configurable)
- Authentication: JWT-based
- API Integration: Semantic Scholar API

## Setup

1. Clone the repository:
```bash
git clone https://github.com/rklorD456/paper-navigator.git
cd paper-navigator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///./papers.db
SEMANTIC_SCHOLAR_API_KEY=your-api-key
SMTP_USERNAME=your-email
SMTP_PASSWORD=your-password
EMAIL_FROM=your-email
```

5. Run the application:
```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

## Environment Variables

- `SECRET_KEY`: Secret key for JWT token generation
- `DATABASE_URL`: Database connection URL
- `SEMANTIC_SCHOLAR_API_KEY`: API key for Semantic Scholar
- `SMTP_USERNAME`: Email username for password reset
- `SMTP_PASSWORD`: Email password for password reset
- `EMAIL_FROM`: Sender email address

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
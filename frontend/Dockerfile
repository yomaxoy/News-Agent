FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy app
COPY . .

# Build
RUN npm run build

# Production image
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production

# Copy built app from builder
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

# Create non-root user
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/api/auth/signin || exit 1

# Run
CMD ["npm", "start"]

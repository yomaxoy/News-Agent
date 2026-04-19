import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          console.error("[AUTH] Missing email or password");
          return null;
        }

        try {
          const apiUrl = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/+$/, "");
          console.log("[AUTH] API URL:", apiUrl);

          const res = await fetch(
            `${apiUrl}/api/auth/login`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                email: credentials.email,
                password: credentials.password,
              }),
            }
          );

          console.log("[AUTH] Backend response status:", res.status);

          if (!res.ok) {
            const errorText = await res.text();
            console.error("[AUTH] Backend error response:", errorText);
            return null;
          }

          const data = await res.json();
          console.log("[AUTH] Backend data keys:", Object.keys(data));

          return {
            id: data.user_id,
            email: data.email,
            token: data.access_token,
          };
        } catch (error: any) {
          console.error("[AUTH] Exception in authorize:", error.message);
          return null;
        }
      },
    }),
  ],
  pages: {
    signIn: "/auth/login",
  },
  callbacks: {
    async jwt({ token, user }: any) {
      if (user) {
        token.id = user.id;
        token.email = user.email;
        token.access_token = user.token;
      }
      return token;
    },
    async session({ session, token }: any) {
      if (session.user) {
        session.user.id = token.id as number;
        (session as any).access_token = token.access_token;
      }
      return session;
    },
  },
  secret: process.env.NEXTAUTH_SECRET,
});

export { handler as GET, handler as POST };
